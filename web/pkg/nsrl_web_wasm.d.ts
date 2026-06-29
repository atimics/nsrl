/* tslint:disable */
/* eslint-disable */

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
    sample_fast(prompt: string, seed: string, candidate_multiplier: number, passes: number, condition_limit: number): SolomonSample;
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
    readonly solomonsampler_sample_fast: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => [number, number, number];
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
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
