class MySeq2SeqTrainer:
    def save_model(self, output_dir=None):
        """手动保存当前模型和分词器"""
        if output_dir is None:
            output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
        tqdm.write(f"模型和分词器已保存到: {output_dir}")
    def __init__(self, model, args, train_dataset=None, eval_dataset=None, tokenizer=None, data_collator=None, compute_metrics=None):
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.data_collator = data_collator
        self.compute_metrics = compute_metrics
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        # 日志平台初始化
        self.report_to = getattr(args, 'report_to', None)
        self._wandb = None
        self._tb_writer = None
        if self.report_to is not None:
            if 'wandb' in self.report_to:
                try:
                    import wandb
                    wandb.init(project=getattr(args, 'wandb_project', 'my_project'), name=getattr(args, 'wandb_run_name', None))
                    self._wandb = wandb
                except ImportError:
                    print('wandb not installed, skipping wandb logging.')
            if 'tensorboard' in self.report_to:
                try:
                    from torch.utils.tensorboard import SummaryWriter
                    self._tb_writer = SummaryWriter(log_dir=getattr(args, 'tb_log_dir', './runs'))
                except ImportError:
                    print('tensorboard not installed, skipping tensorboard logging.')

    def train(self):
        model = self.model
        args = self.args
        # 使用data_collator（如果有）
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=args.train_batch_size,
            shuffle=True,
            collate_fn=self.data_collator if self.data_collator is not None else None
        )
        val_loader = DataLoader(
            self.eval_dataset,
            batch_size=args.eval_batch_size,
            collate_fn=self.data_collator if self.data_collator is not None else None
        ) if self.eval_dataset is not None else None
        use_amp = args.fp16 or args.bf16
        scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
        total_steps = args.num_train_epochs * len(train_loader)
        scheduler = self._create_scheduler(optimizer, total_steps)
        progress_bar = tqdm(total=total_steps, desc="Training", ncols=100)
        global_step = 0
        saved_checkpoints = []
        for epoch in range(args.num_train_epochs):
            epoch_loss = 0
            optimizer.zero_grad()
            for step, batch in enumerate(train_loader):
                model.train()
                # 动态获取主输入名
                input_name = getattr(model, 'main_input_name', 'input_ids')
                model_inputs = {input_name: batch[input_name].to(self.device), 'labels': batch['labels'].to(self.device)}
                with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.bfloat16 if args.bf16 else torch.float16):
                    outputs = model(**model_inputs)
                    loss = outputs.loss / args.gradient_accumulation_steps
                if use_amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                if (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
                    grad_norm = self._compute_grad_norm()
                    if use_amp:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                epoch_loss += loss.item() * args.gradient_accumulation_steps
                global_step += 1
                progress_bar.update(1)
                progress_bar.set_postfix({
                    "loss": f"{loss.item() * args.gradient_accumulation_steps:.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}"
                })
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    current_loss = loss.item() * args.gradient_accumulation_steps
                    # 计算真实epoch进度
                    real_epoch = epoch + (step + 1) / len(train_loader)
                    log_str = (
                        f"[Step {global_step:>5}] | Epoch: {real_epoch:>6.3f} | "
                        f"loss: {current_loss:>8.4f} | grad_norm: {grad_norm:>8.4f} | lr: {scheduler.get_last_lr()[0]:.2e}"
                    )
                    tqdm.write(log_str)
                    # 日志上报
                    if self._wandb is not None:
                        self._wandb.log({
                            'train/loss': current_loss,
                            'train/grad_norm': grad_norm,
                            'train/lr': scheduler.get_last_lr()[0],
                            'epoch': epoch+1,
                            'step': global_step
                        }, step=global_step)
                    if self._tb_writer is not None:
                        self._tb_writer.add_scalar('train/loss', current_loss, global_step)
                        self._tb_writer.add_scalar('train/grad_norm', grad_norm, global_step)
                        self._tb_writer.add_scalar('train/lr', scheduler.get_last_lr()[0], global_step)
                if args.eval_strategy == "steps" and args.eval_steps > 0 and global_step % args.eval_steps == 0 and val_loader is not None:
                    val_result = self.evaluate(val_loader, desc=f"Eval@Step{global_step}")
                    if isinstance(val_result, tuple):
                        val_loss, metrics = val_result
                        metrics_str = ' | '.join([f"{k}: {float(v):.4f}" for k, v in metrics.items()]) if isinstance(metrics, dict) else str(metrics)
                        log_str = (
                            f"[Step {global_step:>5}] | Val loss: {val_loss:>8.4f} | {metrics_str}"
                        )
                        tqdm.write(log_str)
                        # 日志上报
                        if self._wandb is not None:
                            log_dict = {'eval/loss': val_loss, 'epoch': epoch+1, 'step': global_step}
                            if isinstance(metrics, dict):
                                for k, v in metrics.items():
                                    log_dict[f'eval/{k}'] = float(v)
                            self._wandb.log(log_dict, step=global_step)
                        if self._tb_writer is not None:
                            self._tb_writer.add_scalar('eval/loss', val_loss, global_step)
                            if isinstance(metrics, dict):
                                for k, v in metrics.items():
                                    self._tb_writer.add_scalar(f'eval/{k}', float(v), global_step)
                    else:
                        tqdm.write(f"[Step {global_step:>5}] | Val loss: {val_result:>8.4f}")
                        if self._wandb is not None:
                            self._wandb.log({'eval/loss': val_result, 'epoch': epoch+1, 'step': global_step}, step=global_step)
                        if self._tb_writer is not None:
                            self._tb_writer.add_scalar('eval/loss', val_result, global_step)
                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    self._save_checkpoint(global_step, saved_checkpoints)
            avg_loss = epoch_loss / len(train_loader)
            tqdm.write(f"[Epoch {epoch+1}] Train loss: {avg_loss:.4f}")
            if args.eval_strategy == "epoch" and val_loader is not None:
                val_result = self.evaluate(val_loader, desc=f"Epoch {epoch+1}/{args.num_train_epochs} [Val]")
                if isinstance(val_result, tuple):
                    val_loss, metrics = val_result
                    tqdm.write(f"[Epoch {epoch+1}] Val loss: {val_loss:.4f} | metrics: {metrics}")
                else:
                    tqdm.write(f"[Epoch {epoch+1}] Val loss: {val_result:.4f}")
            if args.save_steps == -1:
                self._save_checkpoint(f"epoch{epoch+1}", saved_checkpoints)
        progress_bar.close()
        tqdm.write("训练完成！")

    def evaluate(self, val_loader=None, desc="Eval"):
        if val_loader is None:
            val_loader = DataLoader(self.eval_dataset, batch_size=self.args.eval_batch_size)
        return self._evaluate(val_loader, desc)

    def _save_checkpoint(self, step, saved_checkpoints):
        """保存检查点"""
        path = os.path.join(self.args.output_dir, f"checkpoint-{step}")
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        saved_checkpoints.append(path)
        while self.args.save_total_limit > 0 and len(saved_checkpoints) > self.args.save_total_limit:
            shutil.rmtree(saved_checkpoints.pop(0))

    def _create_scheduler(self, optimizer, total_steps):
        """创建学习率调度器"""
        if self.args.lr_scheduler_type == "linear":
            return LambdaLR(optimizer, lambda s: s / max(1, self.args.warmup_steps) if s < self.args.warmup_steps else max(0.0, (total_steps - s) / max(1, total_steps - self.args.warmup_steps)))
        if self.args.lr_scheduler_type == "cosine":
            return CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=0)
        if self.args.lr_scheduler_type == "constant":
            return LambdaLR(optimizer, lambda _: 1.0)
        raise ValueError(f"Unsupported lr_scheduler_type: {self.args.lr_scheduler_type}")

    def _compute_grad_norm(self):
        """计算模型梯度范数"""
        return sum((p.grad.data.norm(2).item() ** 2 for p in self.model.parameters() if p.grad is not None)) ** 0.5

    def _evaluate(self, val_loader, desc):
        """评估模型性能，compute_metrics需外部传入"""
        self.model.eval()
        val_loss, predictions, references = 0, [], []
        gen_config = getattr(self.model, 'generation_config', None)
        input_name = getattr(self.model, 'main_input_name', 'input_ids')
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=desc, ncols=100, leave=False):
                batch_inputs = {input_name: batch[input_name].to(self.device)}
                lbl = batch["labels"].to(self.device)
                out = self.model(**batch_inputs, labels=lbl)
                val_loss += out.loss.item()
                if hasattr(self.model, 'generate') and gen_config is not None:
                    if input_name == "pixel_values":
                        encoder_outputs = self.model.get_encoder()(pixel_values=batch_inputs[input_name])
                    else:
                        encoder_outputs = self.model.get_encoder()(input_ids=batch_inputs[input_name])
                    preds = self.model.generate(encoder_outputs=encoder_outputs, generation_config=gen_config)
                    predictions.extend(preds.cpu().tolist())
                    references.extend(lbl.cpu().tolist())
        self.model.train()
        if predictions and self.compute_metrics:
            pred = type('Pred', (), {})()
            pred.predictions, pred.label_ids = predictions, references
            return val_loss / len(val_loader), self.compute_metrics(pred)
        return val_loss / len(val_loader)

import os
import shutil
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR

from dataclasses import dataclass, field

@dataclass
class MySeq2SeqTrainingArguments:
    output_dir: str = 'VIT_GPT2_EDM'
    train_batch_size: int = 8
    eval_batch_size: int = 8
    eval_strategy: str = "steps"
    eval_steps: int = 128
    logging_steps: int = 128
    save_steps: int = 2048
    warmup_steps: int = 1024
    learning_rate: float = 5e-5
    num_train_epochs: int = 3
    save_total_limit: int = 1
    lr_scheduler_type: str = "linear"
    gradient_accumulation_steps: int = 1
    fp16: bool = False
    bf16: bool = False
    # 日志平台参数
    report_to: list = field(default_factory=list)  # e.g. ["wandb", "tensorboard"]
    wandb_project: str = 'my_project'
    wandb_run_name: str = None
    tb_log_dir: str = './runs'
