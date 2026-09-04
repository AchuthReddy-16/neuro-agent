/**
 * Regression: composer must clear on send while preserving the question to analyze.
 * Run: npx --yes tsx web/scripts/composer-clear-on-send-check.ts
 */
import assert from "node:assert/strict";
import { consumeComposerDraft } from "../src/lib/composer.ts";

const empty = consumeComposerDraft("   ");
assert.equal(empty, null);

const sent = consumeComposerDraft("  Show PSD for C3  ");
assert.ok(sent);
assert.equal(sent.question, "Show PSD for C3");
assert.equal(sent.nextDraft, "");

const followUp = consumeComposerDraft(sent.nextDraft);
assert.equal(followUp, null, "cleared draft must not re-send");

console.log("ok  composer clears on send; empty/whitespace rejected");
