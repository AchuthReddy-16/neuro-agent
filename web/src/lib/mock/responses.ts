import type {
  AgentAnswer,
  AnalyzeResponse,
  TimelineStage,
} from "@/lib/types";
import {
  DEFAULT_SYSTEM_INFO,
  DEMO_BETA_TOP5,
  DEMO_QUESTION,
  SAMPLE_IMAGES,
  TIMELINE_STAGE_NAMES,
} from "@/lib/constants";
import { resolveApiUrl } from "@/lib/config";

let mockAnswerSeq = 0;
function nextMockAnswerId(): string {
  mockAnswerSeq += 1;
  return `ans-local-${Date.now()}-${mockAnswerSeq}`;
}

function buildTimeline(opts: {
  vision: boolean;
  verifier: boolean;
  recovery: boolean;
  latencies?: Partial<Record<(typeof TIMELINE_STAGE_NAMES)[number], number>>;
}): TimelineStage[] {
  const L = opts.latencies ?? {};
  return [
    {
      id: "t-routing",
      name: "Routing",
      status: "complete",
      latencyMs: L.Routing ?? 12,
      summary: opts.vision ? "VISION + TOOLS" : "TEXT + TOOLS",
    },
    {
      id: "t-tools",
      name: "Tool execution",
      status: "complete",
      latencyMs: L["Tool execution"] ?? 210,
      summary: "Deterministic EEG analysis tools",
    },
    {
      id: "t-vision",
      name: "Vision analysis",
      status: opts.vision ? "complete" : "skipped",
      latencyMs: opts.vision ? (L["Vision analysis"] ?? 340) : undefined,
      summary: opts.vision ? "VLM inspection of selected plots" : "Not required for this question",
    },
    {
      id: "t-evidence",
      name: "Evidence assembly",
      status: "complete",
      latencyMs: L["Evidence assembly"] ?? 45,
    },
    {
      id: "t-synthesis",
      name: "Synthesis",
      status: "complete",
      latencyMs: L.Synthesis ?? 180,
    },
    {
      id: "t-verify",
      name: "Verification",
      status: opts.verifier ? "complete" : "skipped",
      latencyMs: opts.verifier ? (L.Verification ?? 95) : undefined,
      summary: opts.verifier ? "Consistency check triggered" : "Passed without intervention",
    },
    {
      id: "t-recovery",
      name: "Recovery",
      status: opts.recovery ? "complete" : "skipped",
      latencyMs: opts.recovery ? (L.Recovery ?? 120) : undefined,
      summary: opts.recovery ? "Re-ran tools after verification flag" : undefined,
    },
  ];
}

export function analyzeResponseToAgentAnswer(
  res: AnalyzeResponse,
  opts?: { isDemo?: boolean; question?: string },
): AgentAnswer {
  const route = res.route;
  const visualRaw = res.visual_evidence ?? [];
  // Honest TEXT routing: do not present tool-sidecar images as vision evidence
  const visualEvidence =
    route === "VISION"
      ? visualRaw.map((v) => ({
          ...v,
          tab: v.tab as AgentAnswer["visualEvidence"][number]["tab"],
          imageUrl: resolveApiUrl(v.imageUrl) ?? v.imageUrl,
          observation: v.observation ?? v.vlm_interpretation ?? undefined,
        }))
      : [];

  return {
    id: res.id ?? `ans-${Date.now()}`,
    question: opts?.question ?? res.question ?? "",
    answer: res.answer,
    route,
    computedEvidence: res.computed_evidence ?? [],
    visualEvidence,
    modelInterpretation: res.model_interpretation ?? "",
    toolsUsed: res.tools_used ?? [],
    verification: {
      status: (res.verification?.status as AgentAnswer["verification"]["status"]) ?? "skipped",
      message: res.verification?.message ?? undefined,
      recoveryPerformed: res.verification?.recoveryPerformed ?? false,
    },
    uncertainty: res.uncertainty ?? "",
    timing: {
      totalMs: res.timing?.totalMs ?? undefined,
      routingMs: res.timing?.routingMs ?? undefined,
      toolsMs: res.timing?.toolsMs ?? undefined,
      visionMs: res.timing?.visionMs ?? undefined,
      synthesisMs: res.timing?.synthesisMs ?? undefined,
      verificationMs: res.timing?.verificationMs ?? undefined,
    },
    system: {
      textModel: res.system?.textModel ?? "unknown",
      visionModel: res.system?.visionModel ?? "unknown",
      precision: res.system?.precision ?? "INT8 W8A8",
      serving: res.system?.serving ?? "unknown",
      route: res.system?.route ?? route,
      verifierStatus: (res.system?.verifierStatus as AgentAnswer["system"]["verifierStatus"]) ??
        (res.verification?.status as AgentAnswer["system"]["verifierStatus"]),
    },
    timeline: (res.timeline ?? []).map((s) => ({
      id: s.id,
      name: s.name,
      status: s.status as AgentAnswer["timeline"][number]["status"],
      latencyMs: s.latencyMs,
      summary: s.summary,
    })),
    isDemo: opts?.isDemo ?? false,
    rawToolOutput: res.raw_tool_output,
    evidence: res.computed_evidence ?? [],
    visualRefs: visualEvidence,
    routeDetail: res.route_detail,
    selectedImageId: res.sourceImageId ?? res.source_image_id ?? null,
    selectedImageName: res.sourceImageName ?? res.source_image_name ?? null,
    visionUsed: Boolean(res.visionUsed ?? res.vision_used),
    visionAssetOrigin: res.visionAssetOrigin ?? res.vision_asset_origin ?? null,
  };
}

/** Condition-comparison fixture aligned with DEMO_QUESTION / "Compare the two conditions." */
export function createDemoAnalyzeResponse(): AnalyzeResponse {
  const vision = false;
  const timeline = buildTimeline({ vision, verifier: false, recovery: false });
  return {
    id: "ans-demo-001",
    question: DEMO_QUESTION,
    answer:
      "Left-fist imagery shows higher mean beta-band power (0.168) than right-fist imagery (0.145) in this demo session — a +15.9% relative difference. Channel-level contrast is largest at C3 vs C4 (0.184 vs 0.132).",
    route: "TEXT",
    computed_evidence: [
      { label: "Left-fist mean beta", value: "0.168", tool: "Condition Comparison" },
      { label: "Right-fist mean beta", value: "0.145", tool: "Condition Comparison" },
      { label: "Relative difference", value: "+15.9%", highlight: true, tool: "Condition Comparison" },
      { label: "C3 beta (left-fist)", value: "0.184", highlight: true, tool: "Band Power Analysis" },
      { label: "C4 beta (left-fist)", value: "0.132", tool: "Band Power Analysis" },
    ],
    visual_evidence: [],
    model_interpretation:
      "Deterministic condition comparison attributes the contrast to beta-band means already present in the demo fixture; no vision pass was required.",
    tools_used: ["Condition Comparison", "Band Power Analysis"],
    verification: {
      status: "passed",
      message: "Tool outputs consistent; no recovery required.",
      recoveryPerformed: false,
    },
    uncertainty:
      "Moderate confidence. Demo fixture values for a single session; absolute scales may differ across subjects.",
    timing: {
      totalMs: 412,
      routingMs: 12,
      toolsMs: 210,
      synthesisMs: 180,
      verificationMs: 10,
    },
    system: {
      ...DEFAULT_SYSTEM_INFO,
      route: "TEXT",
      verifierStatus: "passed",
    },
    timeline,
    raw_tool_output: JSON.stringify(
      {
        band_power_analysis: {
          band: "beta",
          channels: { C3: 0.184, C4: 0.132, FC3: 0.156, FC4: 0.141 },
        },
        condition_comparison: {
          left_fist: { mean_beta: 0.168 },
          right_fist: { mean_beta: 0.145 },
          delta_pct: 15.9,
        },
      },
      null,
      2,
    ),
  };
}

export function createDemoAgentAnswer(): AgentAnswer {
  const answer = analyzeResponseToAgentAnswer(createDemoAnalyzeResponse(), {
    isDemo: true,
    question: DEMO_QUESTION,
  });
  answer.selectedImageId = null;
  answer.selectedImageName = null;
  return answer;
}

function isBetaChannelRankingQuestion(lower: string): boolean {
  return (
    /highest beta/.test(lower) ||
    /five eeg channels/.test(lower) ||
    /beta-band power/.test(lower) ||
    (/which channels/.test(lower) && /beta/.test(lower))
  );
}

function isConditionCompareQuestion(lower: string): boolean {
  return (
    /compare the two conditions/.test(lower) ||
    /compare beta-band activity between left- and right-fist/.test(lower) ||
    (/compare/.test(lower) && /(condition|left|right|fist)/.test(lower))
  );
}

function isFigureInterpretQuestion(lower: string): boolean {
  return (
    /interpret the selected figure/.test(lower) ||
    /\b(figure|plot|topomap|spectrogram|visual)\b/.test(lower)
  );
}

function betaTop5Fixture(question: string): AnalyzeResponse {
  const ranking = DEMO_BETA_TOP5;
  const channels = ranking.map((r) => r.channel);
  const channelList =
    channels.slice(0, -1).join(", ") + `, and ${channels[channels.length - 1]}`;
  const detail =
    `${channels[0]} is highest at ${ranking[0].betaPowerUV2.toFixed(2)} μV², followed by ` +
    ranking
      .slice(1)
      .map((r) => `${r.channel} at ${r.betaPowerUV2.toFixed(2)} μV²`)
      .join(", ");
  const computed_evidence: AnalyzeResponse["computed_evidence"] = [];
  for (let i = 0; i < ranking.length; i += 1) {
    const r = ranking[i];
    computed_evidence.push({
      label: `Rank ${i + 1} · ${r.channel}`,
      value: String(r.betaPowerUV2),
      unit: "μV²",
      highlight: i === 0,
      tool: "rank_channels_for_sample",
    });
  }

  return {
    id: nextMockAnswerId(),
    question,
    answer: `The five channels with the highest beta-band power are ${channelList}. ${detail}.`,
    route: "TEXT",
    computed_evidence,
    visual_evidence: [],
    model_interpretation:
      "Ranking is the descending beta_power ordering from the deterministic channel-ranking tool for this sample.",
    tools_used: ["rank_channels_for_sample"],
    verification: {
      status: "passed",
      message: "Ranking matches deterministic tool output.",
      recoveryPerformed: false,
    },
    uncertainty:
      "Moderate confidence. Values are sample-specific beta_power (μV²); ranking can change with band definition or preprocessing.",
    timing: {
      totalMs: 412,
      routingMs: 10,
      toolsMs: 240,
      synthesisMs: 140,
      verificationMs: 22,
    },
    system: {
      ...DEFAULT_SYSTEM_INFO,
      route: "TEXT",
      verifierStatus: "passed",
    },
    timeline: buildTimeline({ vision: false, verifier: false, recovery: false }),
    raw_tool_output: JSON.stringify(
      {
        ranking: ranking.map((r) => r.channel),
        values: Object.fromEntries(ranking.map((r) => [r.channel, r.betaPowerUV2])),
        metric: "beta_power",
        units: "uV2",
        top_k: 5,
      },
      null,
      2,
    ),
  };
}

export function createMockAnswer(
  question: string,
  opts?: { selectedImage?: { id: string; name: string; url?: string } | null },
): AgentAnswer {
  const lower = question.toLowerCase();
  const selected = opts?.selectedImage;

  if (isBetaChannelRankingQuestion(lower)) {
    const answer = analyzeResponseToAgentAnswer(betaTop5Fixture(question), {
      isDemo: true,
      question,
    });
    answer.selectedImageId = null;
    answer.selectedImageName = null;
    return answer;
  }

  if (isConditionCompareQuestion(lower)) {
    const base = createDemoAnalyzeResponse();
    base.id = nextMockAnswerId();
    base.question = question;
    const answer = analyzeResponseToAgentAnswer(base, { isDemo: true, question });
    answer.selectedImageId = null;
    answer.selectedImageName = null;
    return answer;
  }

  if (isFigureInterpretQuestion(lower) || lower.includes("alpha") || lower.includes("mu")) {
    const imageLabel = selected?.name ?? "Selected figure";
    const imageUrl = selected?.url ?? SAMPLE_IMAGES.topomap_left;
    const res: AnalyzeResponse = {
      id: nextMockAnswerId(),
      question,
      answer:
        `The selected figure (${imageLabel}) shows EEG-derived spatial/spectral structure consistent with motor-imagery band activity over central electrodes in this demo session.`,
      route: "VISION",
      computed_evidence: [
        { label: "Selected figure", value: imageLabel, tool: "Vision routing" },
        {
          label: "C3 beta (fixture)",
          value: "0.184",
          highlight: true,
          tool: "Band Power Analysis",
        },
        { label: "C4 beta (fixture)", value: "0.132", tool: "Band Power Analysis" },
      ],
      visual_evidence: [
        {
          id: selected?.id ?? "viz-selected",
          label: imageLabel,
          tab: "topomap",
          observation: `Visual inspection of ${imageLabel}: contralateral / spectral structure supports the tool-derived band summary.`,
          imageUrl,
        },
      ],
      model_interpretation:
        "Short visual read of the selected plot only; numeric band values come from the demo fixture tools.",
      tools_used: ["Band Power Analysis"],
      verification: {
        status: "passed",
        recoveryPerformed: false,
      },
      uncertainty:
        "Moderate confidence. Demo fixture values; figure interpretation is sample-labeled.",
      timing: {
        totalMs: 620,
        routingMs: 14,
        toolsMs: 160,
        visionMs: 280,
        synthesisMs: 150,
        verificationMs: 16,
      },
      system: {
        ...DEFAULT_SYSTEM_INFO,
        route: "VISION",
        verifierStatus: "passed",
      },
      timeline: buildTimeline({ vision: true, verifier: false, recovery: false }),
    };
    const answer = analyzeResponseToAgentAnswer(res, { isDemo: true, question });
    answer.selectedImageId = selected?.id ?? null;
    answer.selectedImageName = selected?.name ?? null;
    return answer;
  }

  // Unknown prompt: still return a condition-compare fixture rather than a mismatched ranking answer.
  const demo = createDemoAgentAnswer();
  demo.question = question;
  demo.id = nextMockAnswerId();
  if (selected) {
    demo.selectedImageId = selected.id;
    demo.selectedImageName = selected.name;
  }
  return demo;
}

/** Progressive timeline for analysis animation */
export function progressiveTimeline(
  final: TimelineStage[],
  stepIndex: number,
): TimelineStage[] {
  return final.map((s, idx) => {
    if (idx < stepIndex) return { ...s, status: s.status === "skipped" ? "skipped" : "complete" };
    if (idx === stepIndex) {
      if (s.status === "skipped") return { ...s, status: "skipped" };
      return { ...s, status: "running" };
    }
    return { ...s, status: s.status === "skipped" ? "skipped" : "pending" };
  });
}
