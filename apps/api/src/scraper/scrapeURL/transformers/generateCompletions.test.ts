import { vi } from "vitest";

vi.mock("../../../lib/ai-telemetry", () => ({
  aiTelemetry: (options: unknown) => options,
}));
vi.mock("ai", async importOriginal => {
  const actual = await importOriginal<typeof import("ai")>();
  return {
    ...actual,
    generateObject: vi.fn(async () => ({
      object: { ok: true },
      usage: { inputTokens: 1, outputTokens: 1 },
    })),
  };
});

import { generateObject } from "ai";
import { encoding_for_model } from "@dqbd/tiktoken";
import type { Mock } from "vitest";
import {
  generateCompletions,
  generateCrawlerOptionsFromPrompt,
} from "./llmExtract";
import { CostTracking } from "../../../lib/cost-tracking";
import { config } from "../../../config";
import { modelPrices } from "../../../lib/extract/usage/model-prices";

const noopLogger = {
  warn: () => {},
  info: () => {},
  error: () => {},
  debug: () => {},
  child: () => noopLogger,
} as any;

function run(markdown: string, modelId: string, schema?: any) {
  return generateCompletions({
    logger: noopLogger,
    options: { schema },
    markdown,
    model: { modelId } as any,
    retryModel: { modelId } as any,
    costTrackingOptions: { costTracking: new CostTracking(), metadata: {} },
    metadata: { teamId: "test-team" },
  });
}

function lastCall() {
  const calls = (generateObject as Mock).mock.calls;
  expect(calls).toHaveLength(1);
  return calls[0][0];
}

function countTokens(text: string): number {
  const encoder = encoding_for_model("gpt-4o-mini");
  try {
    return encoder.encode(text).length;
  } finally {
    encoder.free();
  }
}

const originalMaxInputTokens = config.MODEL_MAX_INPUT_TOKENS;

beforeEach(() => {
  (generateObject as Mock).mockClear();
  config.MODEL_MAX_INPUT_TOKENS = undefined;
});

afterEach(() => {
  config.MODEL_MAX_INPUT_TOKENS = originalMaxInputTokens;
});

describe("generateCompletions content trimming", () => {
  // ~3.3 characters per token like real pages: ~400k tokens, three times
  // gpt-4o-mini's window.
  const hugeMarkdown =
    "The quick brown fox jumps over the lazy dog. [link](https://example.com/a/b?c=1) 12345 ".repeat(
      15_000,
    );

  it("trims content that would overflow the model's context window", async () => {
    const result = await run(hugeMarkdown, "gpt-4o-mini");

    expect(result.warning).toContain(
      "the input has been automatically trimmed",
    );
    // The content is trimmed to exactly 80% of the window; the rest of the
    // prompt is a short instruction.
    const budget = Math.floor(
      modelPrices["gpt-4o-mini"].max_input_tokens * 0.8,
    );
    const promptTokens = countTokens(lastCall().prompt);
    expect(promptTokens).toBeGreaterThan(budget);
    expect(promptTokens).toBeLessThanOrEqual(budget + 100);
  });

  it("leaves content that fits untouched", async () => {
    const markdown = "a small page";
    const result = await run(markdown, "gpt-4o-mini");

    expect(result.warning).toBeUndefined();
    expect(lastCall().prompt).toContain(markdown);
  });

  const localModel = "self-hosted-model:latest";
  const pageHead = "The founder is the first fact on the page.";
  const pageTail = "Trailing navigation should not displace the page head.";
  const localMarkdown = [
    pageHead,
    hugeMarkdown.slice(0, 100_000),
    pageTail,
  ].join("\n\n");

  it.each([
    ["default", undefined, 8192],
    ["configured", 2048, 2048],
  ] as const)(
    "preserves extraction instructions and the page head within the %s budget for an unlisted model",
    async (_label, configuredLimit, limit) => {
      config.MODEL_MAX_INPUT_TOKENS = configuredLimit;
      const result = await run(localMarkdown, localModel, {
        type: "object",
        properties: { founder: { type: "string" } },
      });

      const { prompt, schema } = lastCall();
      expect(result.warning).toContain(
        "the input has been automatically trimmed",
      );
      expect(prompt).toMatch(
        /^Transform the following content into structured JSON output/,
      );
      expect(prompt).toContain(pageHead);
      expect(prompt).not.toContain(pageTail);
      expect(schema.jsonSchema.properties).toEqual({
        founder: { type: "string" },
      });
      expect(
        countTokens(prompt) + countTokens(JSON.stringify(schema.jsonSchema)),
      ).toBeLessThan(limit);
    },
  );

  it("does not warn or trim when an unlisted model's input fits", async () => {
    const markdown = "Product details. ".repeat(600);
    const result = await run(markdown, localModel);

    expect(result.warning).toBeUndefined();
    expect(lastCall().prompt).toContain(markdown);
  });

  it("does not use an output-only limit as the input budget", async () => {
    const model = "openrouter/microsoft/wizardlm-2-8x22b:nitro";
    expect("max_input_tokens" in modelPrices[model]).toBe(false);
    const result = await run(hugeMarkdown, model);

    expect(result.warning).toBeUndefined();
    expect(lastCall().prompt).toContain(hugeMarkdown);
  });
});

describe("generateCompletions schema normalization", () => {
  it("turns a bare property map into an object schema, skipping $schema", async () => {
    await run("page", "gpt-4o-mini", {
      $schema: "http://json-schema.org/draft-07/schema#",
      judgments: {
        type: "array",
        items: {
          type: "object",
          properties: { citation: { type: "String" } },
        },
      },
    });

    expect(lastCall().schema.jsonSchema).toEqual({
      type: "object",
      properties: {
        judgments: {
          type: "array",
          items: {
            type: "object",
            properties: { citation: { type: "string" } },
            required: ["citation"],
            additionalProperties: false,
          },
        },
      },
      required: ["judgments"],
      additionalProperties: false,
    });
  });

  it("wraps nullable root arrays in an object", async () => {
    await run("page", "gpt-4o-mini", {
      type: ["array", "null"],
      items: { type: "string" },
    });

    expect(lastCall().schema.jsonSchema).toEqual({
      type: "object",
      properties: {
        items: { type: ["array", "null"], items: { type: "string" } },
      },
      required: ["items"],
      additionalProperties: false,
    });
  });

  it("closes nullable objects so strict mode accepts them", async () => {
    await run("page", "gpt-4o-mini", {
      type: "object",
      properties: {
        address: {
          type: ["object", "null"],
          properties: { city: { type: "string" } },
        },
      },
    });

    expect(lastCall().schema.jsonSchema.properties.address).toEqual({
      type: ["object", "null"],
      properties: { city: { type: "string" } },
      required: ["city"],
      additionalProperties: false,
    });
  });
});

describe("generateCompletions telemetry metadata", () => {
  it("tags crawl-prompt spans with the crawl id", async () => {
    await generateCrawlerOptionsFromPrompt(
      "only the blog",
      noopLogger,
      new CostTracking(),
      { teamId: "test-team", crawlId: "test-crawl" },
    );

    const telemetry = lastCall().telemetry;
    expect(telemetry.functionId).toBe("generateCrawlerOptionsFromPrompt");
    expect(telemetry.metadata).toMatchObject({
      teamId: "test-team",
      crawlId: "test-crawl",
    });
  });

  it("leaves crawlId out when previewing params without a crawl", async () => {
    await generateCrawlerOptionsFromPrompt(
      "only the blog",
      noopLogger,
      new CostTracking(),
      { teamId: "test-team" },
    );

    expect(lastCall().telemetry.metadata).toEqual({
      teamId: "test-team",
    });
  });

  it("keeps tagging scrape spans with the scrape id", async () => {
    await generateCompletions({
      logger: noopLogger,
      options: {},
      markdown: "page",
      costTrackingOptions: { costTracking: new CostTracking(), metadata: {} },
      metadata: { teamId: "test-team", scrapeId: "test-scrape" },
    });

    expect(lastCall().telemetry.metadata).toMatchObject({
      teamId: "test-team",
      scrapeId: "test-scrape",
    });
  });

  it("adds the requesting feature's span metadata without overriding the job ids", async () => {
    await generateCompletions({
      logger: noopLogger,
      options: {},
      markdown: "page",
      costTrackingOptions: { costTracking: new CostTracking(), metadata: {} },
      metadata: {
        teamId: "test-team",
        functionId: "monitor/searchJudge",
        scrapeId: "test-scrape",
        telemetry: {
          teamId: "other-team",
          jobId: "test-check",
          jobKind: "monitor",
          feature: "monitor_search_judge",
        },
      },
    });

    const telemetry = lastCall().telemetry;
    expect(telemetry.functionId).toBe("monitor/searchJudge");
    expect(telemetry.metadata).toMatchObject({
      teamId: "test-team",
      scrapeId: "test-scrape",
      jobId: "test-check",
      jobKind: "monitor",
      feature: "monitor_search_judge",
    });
  });
});
