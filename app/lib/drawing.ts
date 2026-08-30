export type ShapeHint = "round" | "tall" | "wide" | "spiky";

export type DrawingAnalysis = {
  dominantColor: string;
  secondaryColor: string;
  coveragePercent: number;
  aspectRatio: number;
  shapeHint: ShapeHint;
  edgeEnergy: "soft" | "scribbly" | "bold";
  sourceWidth: number;
  sourceHeight: number;
};

export type DrawingExtraction = {
  textureUrl: string;
  previewUrl: string;
  analysis: DrawingAnalysis;
};

type RGB = { r: number; g: number; b: number };

const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value));
const colorDistance = (a: RGB, b: RGB) => Math.sqrt((a.r - b.r) ** 2 + (a.g - b.g) ** 2 + (a.b - b.b) ** 2);
const toHex = ({ r, g, b }: RGB) => `#${[r, g, b].map((value) => clamp(Math.round(value), 0, 255).toString(16).padStart(2, "0")).join("")}`;

function averageBorder(data: Uint8ClampedArray, width: number, height: number): RGB {
  let r = 0;
  let g = 0;
  let b = 0;
  let count = 0;
  const stride = Math.max(1, Math.floor(Math.min(width, height) / 80));

  const take = (x: number, y: number) => {
    const index = (y * width + x) * 4;
    r += data[index];
    g += data[index + 1];
    b += data[index + 2];
    count += 1;
  };

  for (let x = 0; x < width; x += stride) {
    take(x, 0);
    take(x, height - 1);
  }
  for (let y = stride; y < height - stride; y += stride) {
    take(0, y);
    take(width - 1, y);
  }

  return { r: r / count, g: g / count, b: b / count };
}

function foregroundScore(pixel: RGB, background: RGB) {
  const distance = colorDistance(pixel, background);
  const max = Math.max(pixel.r, pixel.g, pixel.b);
  const min = Math.min(pixel.r, pixel.g, pixel.b);
  const saturation = max ? (max - min) / max : 0;
  const darkness = 255 - (pixel.r + pixel.g + pixel.b) / 3;
  return distance + saturation * 68 + darkness * 0.18;
}

function classifyShape(width: number, height: number, coverage: number): ShapeHint {
  const ratio = width / Math.max(1, height);
  if (coverage < 0.12) return "spiky";
  if (ratio > 1.24) return "wide";
  if (ratio < 0.78) return "tall";
  return "round";
}

export function extractDrawingFromVideo(video: HTMLVideoElement): DrawingExtraction {
  if (!video.videoWidth || !video.videoHeight) throw new Error("The camera is still focusing. Try capture again in a moment.");

  const width = 360;
  const height = Math.round(width * (video.videoHeight / video.videoWidth));
  const source = document.createElement("canvas");
  source.width = width;
  source.height = height;
  const context = source.getContext("2d", { willReadFrequently: true });
  if (!context) throw new Error("Canvas processing is unavailable in this browser.");
  context.drawImage(video, 0, 0, width, height);
  const frame = context.getImageData(0, 0, width, height);
  const background = averageBorder(frame.data, width, height);

  let minX = width;
  let minY = height;
  let maxX = 0;
  let maxY = 0;
  let foreground = 0;
  let colorR = 0;
  let colorG = 0;
  let colorB = 0;
  let secondR = 0;
  let secondG = 0;
  let secondB = 0;
  let secondCount = 0;
  const mask = new Uint8Array(width * height);
  const threshold = 66;

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const index = (y * width + x) * 4;
      const pixel = { r: frame.data[index], g: frame.data[index + 1], b: frame.data[index + 2] };
      const score = foregroundScore(pixel, background);
      if (score <= threshold) continue;
      mask[y * width + x] = clamp(Math.round((score - threshold) * 5), 90, 255);
      minX = Math.min(minX, x);
      minY = Math.min(minY, y);
      maxX = Math.max(maxX, x);
      maxY = Math.max(maxY, y);
      foreground += 1;
      colorR += pixel.r;
      colorG += pixel.g;
      colorB += pixel.b;
      if ((x + y) % 3 === 0) {
        secondR += pixel.r;
        secondG += pixel.g;
        secondB += pixel.b;
        secondCount += 1;
      }
    }
  }

  const minimumPixels = width * height * 0.008;
  if (foreground < minimumPixels) throw new Error("I couldn't separate a drawing from the wall. Move closer or use stronger lighting.");

  const padding = Math.max(6, Math.round(Math.max(maxX - minX, maxY - minY) * 0.08));
  minX = clamp(minX - padding, 0, width - 1);
  minY = clamp(minY - padding, 0, height - 1);
  maxX = clamp(maxX + padding, minX + 1, width - 1);
  maxY = clamp(maxY + padding, minY + 1, height - 1);
  const cropWidth = maxX - minX + 1;
  const cropHeight = maxY - minY + 1;

  const output = document.createElement("canvas");
  output.width = 512;
  output.height = 512;
  const outputContext = output.getContext("2d");
  if (!outputContext) throw new Error("Canvas processing is unavailable in this browser.");
  const cropped = context.getImageData(minX, minY, cropWidth, cropHeight);
  const croppedMask = document.createElement("canvas");
  croppedMask.width = cropWidth;
  croppedMask.height = cropHeight;
  const maskContext = croppedMask.getContext("2d");
  if (!maskContext) throw new Error("Canvas processing is unavailable in this browser.");
  const transparent = maskContext.createImageData(cropWidth, cropHeight);

  for (let y = 0; y < cropHeight; y += 1) {
    for (let x = 0; x < cropWidth; x += 1) {
      const from = (y * cropWidth + x) * 4;
      const sourceMaskIndex = (y + minY) * width + x + minX;
      transparent.data[from] = cropped.data[from];
      transparent.data[from + 1] = cropped.data[from + 1];
      transparent.data[from + 2] = cropped.data[from + 2];
      transparent.data[from + 3] = mask[sourceMaskIndex];
    }
  }
  maskContext.putImageData(transparent, 0, 0);

  const scale = Math.min(430 / cropWidth, 430 / cropHeight);
  const drawWidth = cropWidth * scale;
  const drawHeight = cropHeight * scale;
  outputContext.drawImage(croppedMask, (512 - drawWidth) / 2, (512 - drawHeight) / 2, drawWidth, drawHeight);

  const dominant = { r: colorR / foreground, g: colorG / foreground, b: colorB / foreground };
  const secondary = secondCount ? { r: secondR / secondCount, g: secondG / secondCount, b: secondB / secondCount } : background;
  const coverage = foreground / (width * height);

  return {
    textureUrl: output.toDataURL("image/png"),
    previewUrl: source.toDataURL("image/jpeg", 0.78),
    analysis: {
      dominantColor: toHex(dominant),
      secondaryColor: toHex(secondary),
      coveragePercent: Math.round(coverage * 100),
      aspectRatio: Number((cropWidth / cropHeight).toFixed(2)),
      shapeHint: classifyShape(cropWidth, cropHeight, coverage),
      edgeEnergy: coverage < 0.08 ? "scribbly" : coverage > 0.28 ? "bold" : "soft",
      sourceWidth: video.videoWidth,
      sourceHeight: video.videoHeight,
    },
  };
}

export function createDemoDoodle(): DrawingExtraction {
  const output = document.createElement("canvas");
  output.width = 512;
  output.height = 512;
  const context = output.getContext("2d");
  if (!context) throw new Error("Canvas processing is unavailable in this browser.");

  context.lineCap = "round";
  context.lineJoin = "round";
  context.fillStyle = "#ff674d";
  context.strokeStyle = "#18312e";
  context.lineWidth = 15;
  context.beginPath();
  context.moveTo(115, 205);
  context.quadraticCurveTo(92, 92, 196, 82);
  context.quadraticCurveTo(262, 50, 320, 98);
  context.quadraticCurveTo(426, 107, 409, 228);
  context.quadraticCurveTo(432, 356, 318, 405);
  context.quadraticCurveTo(196, 438, 103, 349);
  context.quadraticCurveTo(65, 278, 115, 205);
  context.closePath();
  context.fill();
  context.stroke();

  context.fillStyle = "#5fc7df";
  context.beginPath();
  context.moveTo(135, 132);
  context.lineTo(104, 42);
  context.lineTo(198, 104);
  context.closePath();
  context.fill();
  context.stroke();
  context.beginPath();
  context.moveTo(320, 101);
  context.lineTo(404, 49);
  context.lineTo(374, 144);
  context.closePath();
  context.fill();
  context.stroke();

  const eye = (x: number, y: number) => {
    context.fillStyle = "#fffaf0";
    context.beginPath();
    context.ellipse(x, y, 39, 51, 0, 0, Math.PI * 2);
    context.fill();
    context.stroke();
    context.fillStyle = "#18312e";
    context.beginPath();
    context.arc(x + 8, y + 8, 13, 0, Math.PI * 2);
    context.fill();
  };
  eye(205, 228);
  eye(317, 221);
  context.beginPath();
  context.arc(266, 307, 47, 0.12, Math.PI - 0.12);
  context.stroke();
  context.fillStyle = "#c8f15a";
  context.beginPath();
  context.arc(378, 303, 24, 0, Math.PI * 2);
  context.fill();
  context.stroke();

  return {
    textureUrl: output.toDataURL("image/png"),
    previewUrl: output.toDataURL("image/png"),
    analysis: {
      dominantColor: "#ff674d",
      secondaryColor: "#5fc7df",
      coveragePercent: 31,
      aspectRatio: 1.03,
      shapeHint: "round",
      edgeEnergy: "bold",
      sourceWidth: 512,
      sourceHeight: 512,
    },
  };
}
