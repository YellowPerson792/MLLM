import argparse
from huggingface_hub import snapshot_download
import os

def main():
    parser = argparse.ArgumentParser(description="使用transformers snapshot_download下载模型和分词器")
    parser.add_argument('--repo_id', type=str, required=True, help='模型仓库名，如gpt2或bert-base-uncased')
    parser.add_argument('--local_dir', type=str, default=None, help='保存目录，默认当前目录下repo_id')
    parser.add_argument('--revision', type=str, default=None, help='指定分支/commit/tag')
    args = parser.parse_args()

    if args.local_dir:
        save_dir = args.local_dir
        print(f"正在下载 {args.repo_id} 到 {save_dir} ...")
        snapshot_download(
            repo_id=args.repo_id,
            local_dir=save_dir,
            revision=args.revision,
            local_files_only=False,
            ignore_patterns=None
        )
        print(f"模型 {args.repo_id} 下载完成，已保存到: {save_dir}")
    else:
        print(f"正在下载 {args.repo_id} 到 HuggingFace 默认缓存目录 ...")
        cache_path = snapshot_download(
            repo_id=args.repo_id,
            revision=args.revision,
            local_files_only=False,
            ignore_patterns=None
        )
        print(f"模型 {args.repo_id} 下载完成，已缓存到: {cache_path}")

if __name__ == "__main__":
    main()
