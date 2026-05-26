import torch
import argparse
from pathlib import Path
import json
def export_encoder(checkpoint_path: Path, output_path: Path):
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    encoder_checkpoint = {
        'encoder_state_dict': checkpoint['encoder_state_dict'],
        'model_config': checkpoint['model_config'],
    }
    if 'global_step' in checkpoint:
        encoder_checkpoint['pretrain_steps'] = checkpoint['global_step']
    if 'epoch' in checkpoint:
        encoder_checkpoint['pretrain_epochs'] = checkpoint['epoch']
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(encoder_checkpoint, output_path)
    
    print(f"\nEncoder exported to: {output_path}")
    print(f"Model config: {encoder_checkpoint['model_config']}")
    metadata_path = output_path.parent / f"{output_path.stem}_metadata.json"
    metadata = {
        'source_checkpoint': str(checkpoint_path),
        'export_date': str(Path.ctime(output_path)) if output_path.exists() else None,
        'model_config': encoder_checkpoint['model_config'],
        'pretrain_steps': encoder_checkpoint.get('pretrain_steps'),
        'pretrain_epochs': encoder_checkpoint.get('pretrain_epochs'),
    }
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {metadata_path}")
    print("\nReady for finetuning!")


def main():
    parser = argparse.ArgumentParser(description="Export pretrained encoder for finetuning")
    parser.add_argument( '--checkpoint', type=str, required=True, help='Path to pretraining checkpoint')
    parser.add_argument( '--output', type=str,
        default='./checkpoints/encoder_for_finetuning.pt', help='Output path for encoder checkpoint')
    
    args = parser.parse_args()
    export_encoder(Path(args.checkpoint), Path(args.output))


if __name__ == "__main__":
    main()