# Fine-Tuning `google/gemma-2b-it` for Function Calling

## Project Overview
In this project, we fine-tune the `google/gemma-2b-it` model to enable function-calling capabilities. By default, this model does not support function calling, so our goal is to enhance its ability to interact with external tools and APIs. 

To achieve this, we use a version of the `hermes-function-calling-thinking-V1` dataset and apply supervised fine-tuning (SFT) techniques to improve its performance in function calling tasks.

## Dataset
We utilize the `hermes-function-calling-thinking-V1` dataset, which is designed to train models for structured function calling. The dataset includes examples of function calls, arguments, and expected responses, helping the model generalize function-calling behavior effectively.

## Training Configuration
We use the following hyperparameters and training settings for fine-tuning:

```python
per_device_train_batch_size = 1
per_device_eval_batch_size = 1
gradient_accumulation_steps = 4
logging_steps = 5
learning_rate = 1e-4

max_grad_norm = 1.0
num_train_epochs = 1
warmup_ratio = 0.1
lr_scheduler_type = "cosine"
max_seq_length = 1500

training_arguments = SFTConfig(
    output_dir=output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=per_device_eval_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    save_strategy="no",
    eval_strategy="epoch",
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    max_grad_norm=max_grad_norm,
    weight_decay=0.1,
    warmup_ratio=warmup_ratio,
    lr_scheduler_type=lr_scheduler_type,
    report_to="tensorboard",
    bf16=True,
    hub_private_repo=False,
    push_to_hub=False,
    num_train_epochs=num_train_epochs,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    packing=True,
    max_seq_length=max_seq_length,
)

trainer = SFTTrainer(
    model=model,
    args=training_arguments,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=tokenizer,
    peft_config=peft_config,
)
```

## Model Fine-Tuning Process
1. **Data Preparation**: Preprocessed the `hermes-function-calling-thinking-V1` dataset for training.
2. **Fine-Tuning**: Trained the `google/gemma-2b-it` model using the `SFTTrainer`.
3. **Evaluation**: Evaluated the fine-tuned model on test data to assess its function-calling accuracy.
4. **Logging & Monitoring**: Used TensorBoard for tracking training progress and performance.

## Results & Improvements
- The fine-tuned model now supports structured function calling.
- It can process function call arguments more accurately and generate API-compatible outputs.
- Gradient checkpointing and packing strategies were used to optimize memory efficiency during training.

## Future Work
- Further optimize function-calling performance using reinforcement learning.
- Explore additional datasets to improve generalization.
- Implement real-time function execution testing.

## Installation & Usage
To use the fine-tuned model, follow these steps:


## License
This project is released under the MIT License.
