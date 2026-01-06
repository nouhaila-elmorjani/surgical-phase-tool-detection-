import os

import torch
from PIL import Image
import torchvision.transforms as T

from surgical_phase_tool.config import IMAGE_ROOT, IMAGE_SIZE, DEVICE
from surgical_phase_tool.models.resnet_multitask import PhaseToolNet
from surgical_phase_tool.dataset import PHASE_TO_ID, TOOL_COLUMNS
from surgical_phase_tool.hierarchy.phase_tool_mask import build_phase_tool_mask, apply_phase_mask_to_logits


def load_frames_from_folder(folder: str, max_frames: int = 3):
    image_files = [f for f in sorted(os.listdir(folder)) if f.lower().endswith((".jpg", ".png"))]
    image_files = image_files[:max_frames]

    transform = T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    frames = []
    for fname in image_files:
        img = Image.open(os.path.join(folder, fname)).convert("RGB")
        frames.append(transform(img))

                                    
    if len(frames) == 0:
        raise ValueError("No frames found in folder")
    while len(frames) < 3:
        frames.append(frames[-1])

    frames = frames[:3]

    clip = torch.stack(frames, dim=0)                
    return clip.unsqueeze(0)                   


def run_demo(input_folder: str, checkpoint_path: str):
    print(f"Using device: {DEVICE}")

    clip = load_frames_from_folder(input_folder).to(DEVICE)

    model = PhaseToolNet(backbone_name="resnet18", pretrained=False).to(DEVICE)
    state = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    phase_tool_mask = build_phase_tool_mask().to(DEVICE)

    with torch.no_grad():
        phase_logits, tool_logits = model(clip)
        phase_probs = phase_logits.softmax(dim=-1)

                                
        tool_probs = tool_logits.sigmoid()

                                   
        masked_logits = apply_phase_mask_to_logits(tool_logits, phase_probs, phase_tool_mask, hard=False)
        masked_probs = masked_logits.sigmoid()

    phase_idx = phase_probs.argmax(dim=-1).item()
    phase_name = {v: k for k, v in PHASE_TO_ID.items()}[phase_idx]

    print(f"Predicted phase: {phase_name} (id={phase_idx})")

    print("Top tools (unmasked):")
    probs = tool_probs.squeeze(0).cpu().numpy()
    for i in probs.argsort()[::-1][:5]:
        print(f"  {TOOL_COLUMNS[i]}: {probs[i]:.3f}")

    print("\nTop tools (hierarchy-masked):")
    mprobs = masked_probs.squeeze(0).cpu().numpy()
    for i in mprobs.argsort()[::-1][:5]:
        print(f"  {TOOL_COLUMNS[i]}: {mprobs[i]:.3f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Demo inference on a folder of frames.")
    parser.add_argument("--input_frames", type=str, required=True, help="Folder with frames")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model checkpoint")

    args = parser.parse_args()

    run_demo(args.input_frames, args.checkpoint)
