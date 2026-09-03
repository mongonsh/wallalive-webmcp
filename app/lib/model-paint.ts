export type ModelPaintTool = "brush" | "spray" | "oil" | "spill";
export type ModelPaintBrush = { tool: ModelPaintTool; color: string; size: number };

export type PaintProjection = "texture" | "vertex";

export function resolvePaintProjection(hit: { hasUv: boolean; hasFace: boolean }): PaintProjection | null {
  if (hit.hasUv) return "texture";
  if (hit.hasFace) return "vertex";
  return null;
}

export const PAINT_SOUND_PROFILES: Record<ModelPaintTool, {
  filter: BiquadFilterType;
  frequency: number;
  gain: number;
  loop: boolean;
}> = {
  brush: { filter: "bandpass", frequency: 820, gain: 0.075, loop: true },
  spray: { filter: "highpass", frequency: 1450, gain: 0.13, loop: true },
  oil: { filter: "lowpass", frequency: 620, gain: 0.095, loop: true },
  spill: { filter: "bandpass", frequency: 430, gain: 0.19, loop: false },
};

export type PaintSoundEngine = {
  start: (tool: ModelPaintTool, pressure?: number, size?: number) => boolean;
  update: (pressure?: number, size?: number) => void;
  stop: () => void;
  dispose: () => void;
};

type ActivePaintSound = {
  tool: ModelPaintTool;
  gain: GainNode;
  filter: BiquadFilterNode;
  sources: Array<AudioScheduledSourceNode>;
};

type AudioContextWindow = Window & typeof globalThis & {
  webkitAudioContext?: typeof AudioContext;
};

function browserAudioContext() {
  if (typeof window === "undefined") return null;
  const AudioContextClass = window.AudioContext ?? (window as AudioContextWindow).webkitAudioContext;
  if (!AudioContextClass) return null;
  try {
    return new AudioContextClass();
  } catch {
    return null;
  }
}

function makeNoise(context: AudioContext, seconds: number, decay = false) {
  const length = Math.max(1, Math.floor(context.sampleRate * seconds));
  const buffer = context.createBuffer(1, length, context.sampleRate);
  const data = buffer.getChannelData(0);
  let smoothed = 0;
  for (let index = 0; index < length; index += 1) {
    const white = Math.random() * 2 - 1;
    smoothed = smoothed * 0.72 + white * 0.28;
    const envelope = decay ? Math.pow(1 - index / length, 2.15) : 0.74 + Math.sin(index * 0.017) * 0.05;
    data[index] = (white * 0.54 + smoothed * 0.46) * envelope;
  }
  return buffer;
}

export function createPaintSoundEngine(contextFactory: () => AudioContext | null = browserAudioContext): PaintSoundEngine {
  let context: AudioContext | null = null;
  let active: ActivePaintSound | null = null;

  const getContext = () => {
    context ??= contextFactory();
    if (context?.state === "suspended") void context.resume().catch(() => undefined);
    return context;
  };

  const stop = () => {
    if (!active || !context) return;
    const sound = active;
    active = null;
    const now = context.currentTime;
    sound.gain.gain.cancelScheduledValues(now);
    sound.gain.gain.setValueAtTime(Math.max(0.0001, sound.gain.gain.value), now);
    sound.gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.055);
    sound.sources.forEach((source) => {
      try { source.stop(now + 0.065); } catch { /* already stopped */ }
    });
  };

  const update = (pressure = 0.5, size = 0.42) => {
    if (!active || !context) return;
    const profile = PAINT_SOUND_PROFILES[active.tool];
    const intensity = Math.min(1, Math.max(0.15, pressure || 0.5));
    const width = Math.min(1, Math.max(0.08, size));
    const now = context.currentTime;
    active.gain.gain.setTargetAtTime(profile.gain * (0.52 + intensity * 0.48), now, 0.018);
    active.filter.frequency.setTargetAtTime(profile.frequency * (0.82 + width * 0.36), now, 0.025);
  };

  const start = (tool: ModelPaintTool, pressure = 0.5, size = 0.42) => {
    const audio = getContext();
    if (!audio) return false;
    stop();
    const profile = PAINT_SOUND_PROFILES[tool];
    const now = audio.currentTime;
    const gain = audio.createGain();
    const filter = audio.createBiquadFilter();
    filter.type = profile.filter;
    filter.frequency.setValueAtTime(profile.frequency, now);
    filter.Q.setValueAtTime(tool === "spray" ? 0.38 : tool === "oil" ? 0.72 : 1.05, now);
    gain.gain.setValueAtTime(0.0001, now);
    gain.connect(audio.destination);
    filter.connect(gain);

    const noise = audio.createBufferSource();
    noise.buffer = makeNoise(audio, profile.loop ? 0.32 : 0.72, !profile.loop);
    noise.loop = profile.loop;
    noise.connect(filter);
    const sources: Array<AudioScheduledSourceNode> = [noise];

    if (tool === "oil" || tool === "spill") {
      const tone = audio.createOscillator();
      const toneGain = audio.createGain();
      tone.type = "sine";
      tone.frequency.setValueAtTime(tool === "spill" ? 155 : 86, now);
      tone.frequency.exponentialRampToValueAtTime(tool === "spill" ? 52 : 64, now + (tool === "spill" ? 0.48 : 0.26));
      toneGain.gain.setValueAtTime(tool === "spill" ? 0.075 : 0.018, now);
      if (tool === "spill") toneGain.gain.exponentialRampToValueAtTime(0.0001, now + 0.54);
      tone.connect(toneGain).connect(gain);
      tone.start(now);
      sources.push(tone);
      if (tool === "spill") tone.stop(now + 0.56);
    }

    active = { tool, gain, filter, sources };
    update(pressure, size);
    noise.start(now);
    if (!profile.loop) {
      noise.stop(now + 0.7);
      gain.gain.setValueAtTime(Math.max(0.0001, gain.gain.value), now + 0.12);
      gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.68);
      window.setTimeout(() => { if (active?.gain === gain) active = null; }, 740);
    }
    return true;
  };

  const dispose = () => {
    stop();
    const closing = context;
    context = null;
    if (closing && closing.state !== "closed") void closing.close().catch(() => undefined);
  };

  return { start, update, stop, dispose };
}
