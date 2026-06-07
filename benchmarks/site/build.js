#!/usr/bin/env node

const fs = require("fs");
const path = require("path");

function usage() {
  console.log(`Usage:
  node benchmarks/site/build.js summary --input benchmark.json --metadata metadata.json --out DIR
  node benchmarks/site/build.js site --artifacts DIR --site benchmarks/site --out public
`);
}

function parseArgs(argv) {
  const command = argv[2];
  const args = {};
  for (let i = 3; i < argv.length; i += 1) {
    const arg = argv[i];
    if (!arg.startsWith("--")) {
      throw new Error(`Unexpected argument: ${arg}`);
    }
    const key = arg.slice(2).replace(/-/g, "_");
    const value = argv[i + 1];
    if (!value || value.startsWith("--")) {
      throw new Error(`Missing value for ${arg}`);
    }
    args[key] = value;
    i += 1;
  }
  return { command, args };
}

function readJson(file) {
  return JSON.parse(fs.readFileSync(file, "utf8").replace(/^\uFEFF/, ""));
}

function writeJson(file, value) {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, `${JSON.stringify(value, null, 2)}\n`);
}

function findFiles(root, fileName) {
  if (!fs.existsSync(root)) return [];
  const found = [];
  const stack = [root];
  while (stack.length > 0) {
    const current = stack.pop();
    for (const entry of fs.readdirSync(current, { withFileTypes: true })) {
      const full = path.join(current, entry.name);
      if (entry.isDirectory()) {
        stack.push(full);
      } else if (entry.isFile() && entry.name === fileName) {
        found.push(full);
      }
    }
  }
  return found.sort();
}

function formatNs(ns) {
  if (!Number.isFinite(ns)) return "-";
  if (ns < 1000) return `${ns.toFixed(1)} ns`;
  if (ns < 1_000_000) return `${(ns / 1000).toFixed(2)} us`;
  if (ns < 1_000_000_000) return `${(ns / 1_000_000).toFixed(2)} ms`;
  return `${(ns / 1_000_000_000).toFixed(2)} s`;
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) return "-";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KiB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MiB`;
}

function formatRate(rate) {
  if (!Number.isFinite(rate) || rate <= 0) return "-";
  if (rate >= 1_000_000) return `${(rate / 1_000_000).toFixed(2)} Mops/s`;
  if (rate >= 1_000) return `${(rate / 1_000).toFixed(2)} Kops/s`;
  return `${rate.toFixed(2)} ops/s`;
}

function benchmarkKey(result) {
  return `${result.group}||${result.name}||${result.unit || "op"}`;
}

function geometricMean(values) {
  const clean = values.filter((value) => Number.isFinite(value) && value > 0);
  if (clean.length === 0) return null;
  const logSum = clean.reduce((sum, value) => sum + Math.log(value), 0);
  return Math.exp(logSum / clean.length);
}

function scoreResults(results) {
  const medians = results.map((result) => Number(result.median_ns));
  const geoMedianNs = geometricMean(medians);
  if (!geoMedianNs) return null;
  return 1_000_000_000 / geoMedianNs;
}

function groupScores(results) {
  const grouped = new Map();
  for (const result of results) {
    const group = result.group || "Other";
    if (!grouped.has(group)) grouped.set(group, []);
    grouped.get(group).push(result);
  }
  return Object.fromEntries(
    [...grouped.entries()].map(([group, values]) => [group, scoreResults(values)])
  );
}

function normalizeRun(benchmark, metadata, sourceDir) {
  const results = Array.isArray(benchmark.results) ? benchmark.results : [];
  const sha = metadata.sha || benchmark.sha || "unknown";
  const repository = metadata.repository || "";
  const runId = String(metadata.run_id || path.basename(sourceDir));
  const runUrl = repository && metadata.run_id
    ? `https://github.com/${repository}/actions/runs/${metadata.run_id}`
    : "";
  const commitUrl = repository && sha !== "unknown"
    ? `https://github.com/${repository}/commit/${sha}`
    : "";

  return {
    id: runId,
    repository,
    sha,
    short_sha: metadata.short_sha || sha.slice(0, 7),
    ref_name: metadata.ref_name || metadata.ref || "",
    event: metadata.event || "",
    mode: metadata.mode || benchmark.mode || "",
    workflow: metadata.workflow || "",
    run_id: metadata.run_id || "",
    run_attempt: metadata.run_attempt || "",
    run_number: metadata.run_number || "",
    runner_os: metadata.runner_os || "",
    runner_arch: metadata.runner_arch || "",
    actor: metadata.actor || "",
    created_at: metadata.created_at || benchmark.generated_at || "",
    run_url: runUrl,
    commit_url: commitUrl,
    benchmark: {
      mode: benchmark.mode || metadata.mode || "",
      repeats: benchmark.repeats || 0,
      min_sample_ns: benchmark.min_sample_ns || 0,
      threads: benchmark.threads || 0
    },
    score: scoreResults(results),
    group_scores: groupScores(results),
    case_count: results.length,
    results
  };
}

function loadArtifactRuns(artifactRoot) {
  const benchmarkFiles = findFiles(artifactRoot, "benchmark.json");
  const runs = [];
  for (const benchmarkFile of benchmarkFiles) {
    const dir = path.dirname(benchmarkFile);
    const metadataFile = path.join(dir, "metadata.json");
    try {
      const benchmark = readJson(benchmarkFile);
      const metadata = fs.existsSync(metadataFile) ? readJson(metadataFile) : {};
      runs.push(normalizeRun(benchmark, metadata, dir));
    } catch (error) {
      console.warn(`Skipping ${benchmarkFile}: ${error.message}`);
    }
  }

  const deduped = new Map();
  for (const run of runs) {
    const key = `${run.sha}|${run.mode}|${run.run_id || run.created_at}`;
    const previous = deduped.get(key);
    if (!previous || String(run.created_at) > String(previous.created_at)) {
      deduped.set(key, run);
    }
  }

  return [...deduped.values()].sort((a, b) => String(b.created_at).localeCompare(String(a.created_at)));
}

function makeSummary(benchmark, metadata) {
  const run = normalizeRun(benchmark, metadata, ".");
  const slowest = [...run.results]
    .sort((a, b) => Number(b.median_ns) - Number(a.median_ns))
    .slice(0, 15);
  const allocations = [...run.results]
    .sort((a, b) => Number(b.allocated_bytes) - Number(a.allocated_bytes))
    .slice(0, 10);

  const lines = [];
  lines.push("# FPTM Benchmark Summary");
  lines.push("");
  lines.push(`- Commit: ${run.commit_url ? `[${run.short_sha}](${run.commit_url})` : run.short_sha}`);
  lines.push(`- Mode: \`${run.mode}\``);
  lines.push(`- Cases: ${run.case_count}`);
  lines.push(`- Repeats: ${run.benchmark.repeats}`);
  lines.push(`- Threads: ${run.benchmark.threads}`);
  lines.push(`- Speed score: ${formatRate(run.score)} geometric mean`);
  if (run.run_url) lines.push(`- Workflow run: [${run.run_id}](${run.run_url})`);
  lines.push("");
  lines.push("## Slowest Cases");
  lines.push("");
  lines.push("| Group | Benchmark | Median/op | Allocated |");
  lines.push("|---|---:|---:|---:|");
  for (const result of slowest) {
    lines.push(`| ${result.group} | ${result.name} | ${formatNs(Number(result.median_ns))} | ${formatBytes(Number(result.allocated_bytes))} |`);
  }
  lines.push("");
  lines.push("## Highest Allocation Cases");
  lines.push("");
  lines.push("| Group | Benchmark | Median/op | Allocated |");
  lines.push("|---|---:|---:|---:|");
  for (const result of allocations) {
    lines.push(`| ${result.group} | ${result.name} | ${formatNs(Number(result.median_ns))} | ${formatBytes(Number(result.allocated_bytes))} |`);
  }
  lines.push("");
  return `${lines.join("\n")}\n`;
}

function copySiteFiles(siteDir, outDir) {
  fs.mkdirSync(outDir, { recursive: true });
  for (const file of ["index.html", "styles.css", "app.js"]) {
    fs.copyFileSync(path.join(siteDir, file), path.join(outDir, file));
  }
}

function buildSite(args) {
  const artifactRoot = args.artifacts || ".benchmark-artifacts";
  const siteDir = args.site || path.join("benchmarks", "site");
  const outDir = args.out || "public";
  const runs = loadArtifactRuns(artifactRoot);

  const benchmarkMap = new Map();
  for (const run of runs) {
    for (const result of run.results) {
      const key = benchmarkKey(result);
      if (!benchmarkMap.has(key)) {
        benchmarkMap.set(key, {
          key,
          group: result.group,
          name: result.name,
          unit: result.unit || "op"
        });
      }
    }
  }

  const data = {
    schema: 1,
    generated_at: new Date().toISOString(),
    repository: runs[0]?.repository || "",
    runs,
    benchmarks: [...benchmarkMap.values()].sort((a, b) =>
      `${a.group} ${a.name}`.localeCompare(`${b.group} ${b.name}`)
    )
  };

  fs.rmSync(outDir, { recursive: true, force: true });
  copySiteFiles(siteDir, outDir);
  writeJson(path.join(outDir, "data", "benchmarks.json"), data);
  fs.writeFileSync(
    path.join(outDir, "summary.md"),
    `# Benchmark Dashboard\n\nGenerated ${data.generated_at} from ${runs.length} retained benchmark artifact(s).\n`
  );
  console.log(`Built benchmark dashboard with ${runs.length} run(s).`);
}

function buildSummary(args) {
  if (!args.input || !args.metadata || !args.out) {
    throw new Error("summary requires --input, --metadata, and --out");
  }
  fs.mkdirSync(args.out, { recursive: true });
  const benchmark = readJson(args.input);
  const metadata = readJson(args.metadata);
  const summary = makeSummary(benchmark, metadata);
  fs.writeFileSync(path.join(args.out, "summary.md"), summary);
  writeJson(path.join(args.out, "summary.json"), normalizeRun(benchmark, metadata, "."));
}

function main() {
  const { command, args } = parseArgs(process.argv);
  if (!command || command === "--help" || command === "-h") {
    usage();
    return;
  }
  if (command === "summary") {
    buildSummary(args);
  } else if (command === "site") {
    buildSite(args);
  } else {
    throw new Error(`Unknown command: ${command}`);
  }
}

main();
