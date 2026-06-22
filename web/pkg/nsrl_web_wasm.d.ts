/* tslint:disable */
/* eslint-disable */

export class NsrlChat {
    free(): void;
    [Symbol.dispose](): void;
    adapt_and_reply(history: string, latest_message: string, max_new_tokens: number, sample_seed: number, top_k: number, fine_tune_max_windows: number): string;
    export_model(): Uint8Array;
    import_model(model_bytes: Uint8Array): void;
    model_card(): string;
    constructor(model_bytes: Uint8Array, vocab_tsv: string, token_bytes: Uint8Array);
    reply(prompt: string, max_new_tokens: number, sample_seed: number, top_k: number): string;
}

export class SolomonSample {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    height(): number;
    metadata_json(): string;
    rgba(): Uint8Array;
    width(): number;
}

export class SolomonSampler {
    free(): void;
    [Symbol.dispose](): void;
    model_card(): string;
    constructor(model_bytes: Uint8Array, text_index_tsv: string);
    sample(prompt: string, seed: string, candidate_multiplier: number, passes: number): SolomonSample;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_solomonsample_free: (a: number, b: number) => void;
    readonly __wbg_solomonsampler_free: (a: number, b: number) => void;
    readonly solomonsample_height: (a: number) => number;
    readonly solomonsample_metadata_json: (a: number) => [number, number];
    readonly solomonsample_rgba: (a: number) => [number, number];
    readonly solomonsample_width: (a: number) => number;
    readonly solomonsampler_model_card: (a: number) => [number, number];
    readonly solomonsampler_new: (a: number, b: number, c: number, d: number) => [number, number, number];
    readonly solomonsampler_sample: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => [number, number, number];
    readonly __wbg_nsrlchat_free: (a: number, b: number) => void;
    readonly nsrlchat_adapt_and_reply: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number) => [number, number, number, number];
    readonly nsrlchat_export_model: (a: number) => [number, number, number, number];
    readonly nsrlchat_import_model: (a: number, b: number, c: number) => [number, number];
    readonly nsrlchat_model_card: (a: number) => [number, number];
    readonly nsrlchat_new: (a: number, b: number, c: number, d: number, e: number, f: number) => [number, number, number];
    readonly nsrlchat_reply: (a: number, b: number, c: number, d: number, e: number, f: number) => [number, number, number, number];
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
