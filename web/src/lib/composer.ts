/**
 * Composer send helper — clear draft immediately on successful Send intent.
 * Does not touch conversation history, experiment, or attachments.
 */
export function consumeComposerDraft(
  draft: string,
): { question: string; nextDraft: "" } | null {
  const question = draft.trim();
  if (!question) return null;
  return { question, nextDraft: "" };
}
