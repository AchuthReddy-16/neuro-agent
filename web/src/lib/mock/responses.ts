import type {
  AgentAnswer,
  AnalyzeResponse,
  TimelineStage,
  VisualEvidenceItem,
} from "@/lib/types";
import {
  DEFAULT_SYSTEM_INFO,
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

function demoVisualEvidence(): VisualEvidenceItem[] {
  return [
    {
      id: "viz-topomap-01",
      label: "Topomap #01 — Left Fist (Beta)",
      tab: "topomap",
      observation:
        "Contralateral sensorimotor focus visible over left central electrodes for left-fist imagery.",
      imageUrl: SAMPLE_IMAGES.topomap_left,
    },
    {
      id: "viz-psd-02",
      label: "PSD #02 — Right Fist",
      tab: "psd",
      observation: "Beta-band peak structure differs between C3 and C4 traces.",
      imageUrl: SAMPLE_IMAGES.psd_right,
    },
    {
      id: "viz-compare-01",
      label: "Comparison #01",
      tab: "comparison",
      observation: "Condition contrast plot highlights lateralized beta power differences.",
      imageUrl: SAMPLE_IMAGES.comparison,
    },
  ];
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
  };
}

export function createDemoAnalyzeResponse(): AnalyzeResponse {
  const vision = true;
  const timeline = buildTimeline({ vision, verifier: false, recovery: false });
  return {
    id: "ans-demo-001",
    question: DEMO_QUESTION,
    answer:
      "Beta-band (13–30 Hz) power over sensorimotor cortex is lateralized toward the contralateral hemisphere during motor imagery. For left-fist imagery, C3 (left motor cortex) shows elevated beta desynchronization relative to C4, consistent with event-related desynchronization patterns in BCI motor imagery paradigms.",
    route: "VISION",
    computed_evidence: [
      { label: "Highest beta-power channel", value: "C3", highlight: true, tool: "Channel Ranking" },
      { label: "C3 beta power", value: "0.184", tool: "Band Power Analysis" },
      { label: "C4 beta power", value: "0.132", tool: "Band Power Analysis" },
      { label: "Difference", value: "+39.4%", highlight: true, tool: "Condition Comparison" },
      { label: "Effect size (Cohen's d)", value: "0.87", tool: "Effect Size" },
      { label: "Discriminative rank", value: "C3 > FC3 > CP3", tool: "Channel Ranking" },
    ],
    visual_evidence: demoVisualEvidence(),
    model_interpretation:
      "Taken together, deterministic band-power statistics and plot-level observations support contralateral beta modulation during left-fist motor imagery in this demo session.",
    tools_used: ["Band Power Analysis", "Condition Comparison", "Channel Ranking"],
    verification: {
      status: "passed",
      message: "Tool outputs consistent with visual evidence; no recovery required.",
      recoveryPerformed: false,
    },
    uncertainty:
      "Moderate confidence. Cross-subject EEG variability and electrode impedance differences may affect absolute power estimates. Single-trial imagery timing was not verified.",
    timing: {
      totalMs: 787,
      routingMs: 12,
      toolsMs: 210,
      visionMs: 340,
      synthesisMs: 180,
      verificationMs: 45,
    },
    system: {
      ...DEFAULT_SYSTEM_INFO,
      route: "VISION",
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
        channel_ranking: ["C3", "FC3", "CP3", "C4", "FC4"],
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
  answer.selectedImageId = "img-topo-demo";
  answer.selectedImageName = "beta_topomap.png";
  return answer;
}

export function createMockAnswer(
  question: string,
  opts?: { selectedImage?: { id: string; name: string; url?: string } | null },
): AgentAnswer {
  const lower = question.toLowerCase();
  const selected = opts?.selectedImage;

  if (lower.includes("discriminative") || lower.includes("channel") || lower.includes("highest beta")) {
    const res: AnalyzeResponse = {
      id: nextMockAnswerId(),
      question,
      answer:
        "Sensorimotor channels C3, FC3, and CP3 show the strongest discriminability between left- and right-fist imagery conditions, with C3 ranking highest for beta-band lateralization.",
      route: "TEXT",
      computed_evidence: [
        { label: "Top discriminative channel", value: "C3", highlight: true, tool: "Channel Ranking" },
        { label: "C3 vs C4 separation", value: "0.052", unit: "μV²/Hz", tool: "Channel Ranking" },
        { label: "Classifier accuracy (sample)", value: "78.4%", tool: "Classifier" },
      ],
      visual_evidence: [],
      model_interpretation:
        "Ranking is driven by deterministic channel-separation metrics; no vision pass was required for this question.",
      tools_used: ["Channel Ranking", "Classifier"],
      verification: {
        status: "passed",
        message: "Ranking stable across tool re-check.",
        recoveryPerformed: false,
      },
      uncertainty: "Moderate confidence. Ranking based on held-out validation split (demo fixture).",
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
    };
    const answer = analyzeResponseToAgentAnswer(res, { isDemo: true, question });
    answer.selectedImageId = null;
    answer.selectedImageName = null;
    return answer;
  }

  if (lower.includes("alpha") || lower.includes("mu") || lower.includes("topomap") || lower.includes("figure") || lower.includes("plot") || lower.includes("spectrogram") || lower.includes("visual")) {
    const imageLabel = selected?.name ?? "Selected figure";
    const imageUrl = selected?.url ?? SAMPLE_IMAGES.spectrogram;
    const res: AnalyzeResponse = {
      id: nextMockAnswerId(),
      question,
      answer:
        "Visual inspection of the selected figure, together with band-power tools, indicates prominent alpha/mu or beta-structure patterns over central electrodes consistent with motor imagery.",
      route: "VISION",
      computed_evidence: [
        { label: "Strongest alpha/mu change", value: "C3", highlight: true, tool: "Band Power Analysis" },
        { label: "Alpha suppression", value: "-24.6%", tool: "Band Power Analysis" },
        { label: "Selected figure", value: imageLabel, tool: "Vision routing" },
      ],
      visual_evidence: [
        {
          id: selected?.id ?? "viz-selected",
          label: imageLabel,
          tab: "spectrogram",
          observation: `Vision model inspected ${imageLabel}: spatial/spectral structure supports the tool-derived summary.`,
          imageUrl,
        },
      ],
      model_interpretation:
        "Figure-level observations align with deterministic band metrics for this demo recording.",
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
