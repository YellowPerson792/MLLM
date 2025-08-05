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

    def train(self, resume_from_checkpoint=None, model_already_loaded=False):
        """
        训练函数
        Args:
            resume_from_checkpoint: checkpoint路径
            model_already_loaded: 如果为True，表示模型权重已经预先加载（如通过PeftModel.from_pretrained），
                                只需要恢复训练状态，不再加载模型权重
        """
        args = self.args
        # 使用data_collator（如果有）
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=args.train_batch_size,
            shuffle=False,
            collate_fn=self.data_collator if self.data_collator is not None else None
        )
        val_loader = DataLoader(
            self.eval_dataset,
            batch_size=args.eval_batch_size,
            collate_fn=self.data_collator if self.data_collator is not None else None
        ) if self.eval_dataset is not None else None
        
        # ====== checkpoint 恢复逻辑（必须在创建optimizer之前） ======
        global_step = 0  # batch step计数
        optimizer_step = 0  # optimizer step计数
        start_epoch = 0
        start_step_in_epoch = 0
        
        if resume_from_checkpoint is not None:
            checkpoint_dir = resume_from_checkpoint
            print(f"尝试从 checkpoint 恢复: {checkpoint_dir}")
            
            if not model_already_loaded:
                # 模型权重还未加载，需要进行完整的模型恢复
                print("执行完整的模型权重恢复...")
                
                # 保存原始模型的重要配置
                was_gradient_checkpointing = getattr(self.model, 'gradient_checkpointing', False)
                
                # 检查是否是 PEFT 模型
                is_peft_model = False
                try:
                    from peft import PeftModel
                    is_peft_model = isinstance(self.model, PeftModel)
                    if is_peft_model:
                        print("检测到 PEFT 模型")
                except ImportError:
                    pass
                
                # 恢复模型权重 - 采用原地更新而不是重新创建
                try:
                    if is_peft_model:
                        # PEFT 模型：使用特殊的恢复方式
                        print("使用 PEFT 模型恢复方式")
                        base_model = self.model.get_base_model()
                        self.model = PeftModel.from_pretrained(base_model, checkpoint_dir)
                    else:
                        # 普通模型：尝试原地加载权重
                        print("尝试原地加载模型权重...")
                        # 优先尝试加载 safetensors 格式
                        weight_files = ["pytorch_model.safetensors", "pytorch_model.bin"]
                        loaded = False
                        
                        for weight_file in weight_files:
                            weight_path = os.path.join(checkpoint_dir, weight_file)
                            if os.path.exists(weight_path):
                                try:
                                    print(f"加载权重文件: {weight_file}")
                                    if weight_file.endswith('.safetensors'):
                                        from safetensors.torch import load_file
                                        state_dict = load_file(weight_path)
                                    else:
                                        state_dict = torch.load(weight_path, map_location=self.device)
                                    
                                    # 原地加载权重，保持模型结构不变
                                    missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
                                    if missing_keys:
                                        print(f"缺失的键: {len(missing_keys)} 个")
                                    if unexpected_keys:
                                        print(f"意外的键: {len(unexpected_keys)} 个")
                                    print(f"✓ 成功原地加载权重: {weight_file}")
                                    loaded = True
                                    break
                                except Exception as e:
                                    print(f"加载 {weight_file} 失败: {e}")
                                    continue
                        
                        if not loaded:
                            print("原地加载失败，尝试 from_pretrained 方式...")
                            self.model = self.model.from_pretrained(checkpoint_dir)
                            print("⚠️ 使用了 from_pretrained，可能需要重新配置模型")
                            
                except Exception as e:
                    print(f"模型恢复失败: {e}")
                    print("跳过模型恢复，使用原始模型权重")
                
                # 确保模型在正确设备上
                self.model.to(self.device)
                
                # 恢复梯度检查点设置
                if was_gradient_checkpointing and hasattr(self.model, 'gradient_checkpointing_enable'):
                    print("重新启用梯度检查点")
                    self.model.gradient_checkpointing_enable()
                
                # 恢复分词器
                if self.tokenizer is not None:
                    try:
                        self.tokenizer = self.tokenizer.from_pretrained(checkpoint_dir)
                        print("✓ 恢复分词器")
                    except:
                        print("分词器恢复失败，使用原始分词器")
                
                print(f"✓ 模型恢复完成，设备: {self.model.device}")
                print(f"✓ 模型类型: {type(self.model)}")
                if hasattr(self.model, 'gradient_checkpointing'):
                    print(f"✓ 梯度检查点: {self.model.gradient_checkpointing}")
            else:
                # 模型权重已经预先加载，跳过模型恢复，只恢复训练状态
                print("✓ 模型权重已预先加载，跳过模型恢复步骤")
                print(f"✓ 当前模型设备: {self.model.device}")
                print(f"✓ 当前模型类型: {type(self.model)}")
                if hasattr(self.model, 'gradient_checkpointing'):
                    print(f"✓ 梯度检查点状态: {self.model.gradient_checkpointing}")
            
            # 恢复训练状态计数器（无论模型是否预加载都需要恢复）
            state_path = os.path.join(checkpoint_dir, "trainer_state.pt")
            if os.path.exists(state_path):
                state = torch.load(state_path, map_location="cpu")
                global_step = state.get("global_step", 0)
                optimizer_step = state.get("optimizer_step", 0)
                start_epoch = state.get("epoch", 0)
                start_step_in_epoch = state.get("step_in_epoch", 0)
                print(f"✓ 恢复计数: global_step={global_step}, optimizer_step={optimizer_step}, epoch={start_epoch}, step_in_epoch={start_step_in_epoch}")
            else:
                print("未检测到 trainer_state.pt，计数器使用初始状态")
        
        # 在模型恢复后创建 optimizer/scheduler/scaler（重要！）
        use_amp = args.fp16 or args.bf16
        scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
        optimizer = optim.AdamW(self.model.parameters(), lr=args.learning_rate)
        # 计算真实的optimizer step总数（考虑梯度累计）
        total_batch_steps = args.num_train_epochs * len(train_loader)
        total_optimizer_steps = (total_batch_steps + args.gradient_accumulation_steps - 1) // args.gradient_accumulation_steps
        scheduler = self._create_scheduler(optimizer, total_optimizer_steps)
        progress_bar = tqdm(total=total_batch_steps, desc="Training", ncols=100)
        saved_checkpoints = []

        # 恢复 optimizer/scheduler/scaler 状态（在创建后）
        if resume_from_checkpoint is not None:
            checkpoint_dir = resume_from_checkpoint
            # 恢复 optimizer/scheduler/scaler 状态
            opt_path = os.path.join(checkpoint_dir, "optimizer.pt")
            sch_path = os.path.join(checkpoint_dir, "scheduler.pt")
            scaler_path = os.path.join(checkpoint_dir, "scaler.pt")
            if os.path.exists(opt_path):
                optimizer.load_state_dict(torch.load(opt_path, map_location=self.device))
                print("✓ 恢复 optimizer 状态")
            else:
                print("未检测到 optimizer.pt，optimizer 使用初始状态")
            if os.path.exists(sch_path):
                scheduler.load_state_dict(torch.load(sch_path, map_location=self.device))
                print("✓ 恢复 scheduler 状态")
            else:
                print("未检测到 scheduler.pt，scheduler 使用初始状态")
            if os.path.exists(scaler_path):
                scaler.load_state_dict(torch.load(scaler_path, map_location=self.device))
                print("✓ 恢复 scaler 状态")
            else:
                print("未检测到 scaler.pt，scaler 使用初始状态")
            
            # 恢复计数器
            state_path = os.path.join(checkpoint_dir, "trainer_state.pt")
            if os.path.exists(state_path):
                state = torch.load(state_path, map_location="cpu")
                global_step = state.get("global_step", 0)
                optimizer_step = state.get("optimizer_step", 0)
                start_epoch = state.get("epoch", 0)
                start_step_in_epoch = state.get("step_in_epoch", 0)
                print(f"✓ 恢复计数: global_step={global_step}, optimizer_step={optimizer_step}, epoch={start_epoch}, step_in_epoch={start_step_in_epoch}")
            else:
                print("未检测到 trainer_state.pt，计数器使用初始状态")
            # 进度条同步
            progress_bar.n = global_step
            progress_bar.last_print_n = global_step
            progress_bar.refresh()

        for epoch in range(start_epoch, args.num_train_epochs):
            epoch_loss = 0
            optimizer.zero_grad()
            loss_accumulated = 0.0
            for step, batch in enumerate(train_loader):
                # 跳过已完成的 step（仅在恢复时生效）
                if epoch == start_epoch and step < start_step_in_epoch:
                    continue
                self.model.train()
                # 动态获取主输入名
                input_name = getattr(self.model, 'main_input_name', 'input_ids')
                model_inputs = {input_name: batch[input_name].to(self.device), 'labels': batch['labels'].to(self.device)}
                # attention_mask支持
                if 'attention_mask' in batch:
                    model_inputs['attention_mask'] = batch['attention_mask'].to(self.device)
                with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.bfloat16 if args.bf16 else torch.float16):
                    outputs = self.model(**model_inputs)
                    loss = outputs.loss / args.gradient_accumulation_steps
                if use_amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                loss_accumulated += loss.item()
                if (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
                    grad_norm = self._compute_grad_norm()
                    if use_amp:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    optimizer_step += 1  # 真正的optimizer step计数
                epoch_loss += loss.item() * args.gradient_accumulation_steps
                global_step += 1  # batch step计数
                progress_bar.update(1)
                real_epoch = epoch + (step + 1) / len(train_loader)
                progress_bar.set_postfix({
                    "ep": f"{real_epoch:.2f}/{args.num_train_epochs}",
                    "step": global_step,
                    "loss": f"{loss.item() * args.gradient_accumulation_steps:.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}"
                })
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    current_loss = loss_accumulated
                    # 计算真实epoch进度
                    real_epoch = epoch + (step + 1) / len(train_loader)
                    log_str = (
                        f"[Batch {global_step:>5}] [Opt {optimizer_step:>4}] [Ep {real_epoch:>6.3f}] | "
                        f"Loss: {current_loss:>7.4f} | GradNorm: {grad_norm:>7.3f} | LR: {scheduler.get_last_lr()[0]:.2e}"
                    )
                    tqdm.write(log_str)
                    # 日志上报 (按HF标准格式)
                    if self._wandb is not None:
                        self._wandb.log({
                            'train/loss': current_loss,
                            'train/grad_norm': grad_norm,
                            'train/learning_rate': scheduler.get_last_lr()[0],
                            'train/epoch': real_epoch,
                            'train/global_step': global_step
                        }, step=global_step)
                    if self._tb_writer is not None:
                        self._tb_writer.add_scalar('train/loss', current_loss, global_step)
                        self._tb_writer.add_scalar('train/grad_norm', grad_norm, global_step)
                        self._tb_writer.add_scalar('train/learning_rate', scheduler.get_last_lr()[0], global_step)
                if (step + 1) % args.gradient_accumulation_steps == 0: 
                        loss_accumulated = 0.0
                if args.eval_strategy == "steps" and args.eval_steps > 0 and global_step % args.eval_steps == 0 and val_loader is not None:
                    val_result = self.evaluate(val_loader, desc=f"Eval@Step{global_step}")
                    if isinstance(val_result, tuple):
                        val_loss, metrics = val_result
                        metrics_str = ' | '.join([f"{k}: {float(v):.4f}" for k, v in metrics.items()]) if isinstance(metrics, dict) else str(metrics)
                        log_str = (
                            f"[Batch {global_step:>5}] [EVAL] | Loss: {val_loss:>7.4f} | {metrics_str}"
                        )
                        tqdm.write(log_str)
                        # 日志上报 (按HF标准格式)
                        if self._wandb is not None:
                            log_dict = {'eval/loss': val_loss, 'train/epoch': real_epoch, 'train/global_step': global_step}
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
                        tqdm.write(f"[Batch {global_step:>5}] [EVAL] | Loss: {val_result:>7.4f}")
                        if self._wandb is not None:
                            self._wandb.log({'eval/loss': val_result, 'train/epoch': real_epoch, 'train/global_step': global_step}, step=global_step)
                        if self._tb_writer is not None:
                            self._tb_writer.add_scalar('eval/loss', val_result, global_step)
                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    tqdm.write(f"[Batch {global_step:>5}] [SAVE] | 保存检查点到 checkpoint-{global_step}")
                    self._save_checkpoint(global_step, saved_checkpoints, optimizer, scheduler, scaler, epoch, step)
            avg_loss = epoch_loss / len(train_loader)
            tqdm.write(f"=== [EPOCH {epoch+1}/{args.num_train_epochs} 完成] | 平均Loss: {avg_loss:.4f} | 总Batch步数: {global_step} | 总Opt步数: {optimizer_step} ===")
            if args.eval_strategy == "epoch" and val_loader is not None:
                val_result = self.evaluate(val_loader, desc=f"Epoch {epoch+1}/{args.num_train_epochs} [Val]")
                if isinstance(val_result, tuple):
                    val_loss, metrics = val_result
                    tqdm.write(f"[EPOCH {epoch+1}] [EVAL] | Loss: {val_loss:.4f} | Metrics: {metrics}")
                else:
                    tqdm.write(f"[EPOCH {epoch+1}] [EVAL] | Loss: {val_result:.4f}")
            if args.save_steps == -1:
                tqdm.write(f"[EPOCH {epoch+1}] [SAVE] | 保存epoch检查点到 checkpoint-epoch{epoch+1}")
                self._save_checkpoint(f"epoch{epoch+1}", saved_checkpoints, optimizer, scheduler, scaler, epoch, 0)
        progress_bar.close()
        tqdm.write("=" * 80)
        tqdm.write(f"🎉 训练完成！总计 {args.num_train_epochs} 个epoch，{global_step} 个batch步数，{optimizer_step} 个优化器步数")
        tqdm.write("=" * 80)
        
    def evaluate(self, val_loader, desc):
        """评估模型性能，compute_metrics需外部传入"""
        self.model.eval()
        val_loss, predictions, references = 0, [], []
        gen_config = getattr(self.model, 'generation_config', None)
        input_name = getattr(self.model, 'main_input_name', 'input_ids')
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=desc, ncols=100, leave=False):
                batch_inputs = {input_name: batch[input_name].to(self.device)}
                if 'attention_mask' in batch:
                    batch_inputs['attention_mask'] = batch['attention_mask'].to(self.device)
                lbl = batch["labels"].to(self.device)
                out = self.model(**batch_inputs, labels=lbl)
                val_loss += out.loss.item()
                if hasattr(self.model, 'generate') and gen_config is not None:
                    encoder_outputs = self.model.get_encoder()(**{input_name: batch_inputs[input_name]})
                    preds = self.model.generate(encoder_outputs=encoder_outputs, generation_config=gen_config)
                    predictions.extend(preds.cpu().tolist())
                    references.extend(lbl.cpu().tolist())
        self.model.train()
        if predictions and self.compute_metrics:
            pred = type('Pred', (), {})()
            pred.predictions, pred.label_ids = predictions, references
            return val_loss / len(val_loader), self.compute_metrics(pred)
        return val_loss / len(val_loader)

    def _save_checkpoint(self, step, saved_checkpoints, optimizer=None, scheduler=None, scaler=None, epoch=0, step_in_epoch=0):
        """保存检查点，包括模型、分词器、optimizer、scheduler、scaler、计数器"""
        path = os.path.join(self.args.output_dir, f"checkpoint-{step}")
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(path)
        # 保存 optimizer/scheduler/scaler 状态
        if optimizer is not None:
            torch.save(optimizer.state_dict(), os.path.join(path, "optimizer.pt"))
        if scheduler is not None:
            torch.save(scheduler.state_dict(), os.path.join(path, "scheduler.pt"))
        if scaler is not None:
            torch.save(scaler.state_dict(), os.path.join(path, "scaler.pt"))
        # 保存计数器
        state = {
            "global_step": getattr(self, "global_step", 0),
            "optimizer_step": getattr(self, "optimizer_step", 0),
            "epoch": epoch,
            "step_in_epoch": step_in_epoch
        }
        torch.save(state, os.path.join(path, "trainer_state.pt"))
        saved_checkpoints.append(path)
        while self.args.save_total_limit > 0 and len(saved_checkpoints) > self.args.save_total_limit:
            shutil.rmtree(saved_checkpoints.pop(0))

    def _create_scheduler(self, optimizer, total_optimizer_steps):
        """创建学习率调度器"""
        # 将warmup_steps从batch step转为optimizer step（如果需要）
        warmup_optimizer_steps = self.args.warmup_steps // self.args.gradient_accumulation_steps if self.args.warmup_steps > 0 else 0
        
        if self.args.lr_scheduler_type == "linear":
            return LambdaLR(optimizer, lambda s: s / max(1, warmup_optimizer_steps) if s < warmup_optimizer_steps else max(0.0, (total_optimizer_steps - s) / max(1, total_optimizer_steps - warmup_optimizer_steps)))
        if self.args.lr_scheduler_type == "cosine":
            return CosineAnnealingLR(optimizer, T_max=total_optimizer_steps, eta_min=0)
        if self.args.lr_scheduler_type == "constant":
            return LambdaLR(optimizer, lambda _: 1.0)
        raise ValueError(f"Unsupported lr_scheduler_type: {self.args.lr_scheduler_type}")

    def _compute_grad_norm(self):
        """计算模型梯度范数"""
        return sum((p.grad.data.norm(2).item() ** 2 for p in self.model.parameters() if p.grad is not None)) ** 0.5

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
