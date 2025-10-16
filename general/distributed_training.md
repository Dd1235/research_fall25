# Distributed Training

- Data parallelism: distributed dataset
- Model parallelism: distributed model, each layer on a different device

## Gradient accumulation

- computational graph: the graph that represents the operations performed on tensors
- after loss.backward(), you don't update parameters each time, you might update every two items(batches/samples)
- Can accumulate also when you want to simulate a larger batch size than your hardware can handle

## Distributed Data Parallel (DDP)

- multi server, single gpu, or single server multi gpu, or multi server multi gpu
- node = gpu
- models weights are intialize to one node, broadcast
- every few batches, all reduce
- update using the gradients received
  go back to training the model

- what happens if one node crashes?
  - easy way: start entire cluster again, lose all progress
  - checkpointing: save model on shared disk every few iterations
  - initialize weights from last checkpoint
  - who should save the checkpoint?
    - check rank of the node

Talk is cheap. Show me the code - Linus Torvalds

- Paperspace is a good cloud service.
- create account and set up ssh keys
- create a private network. ml in a box, about $5 for running.
- use weights and biases to keep track of the experiments

local rank vs rank

- among local computer vs over the entire cluster

- in dataloader use DistributedSampler to ensure each node gets different data and don't add shuffle in the dataloader
- path indicates checkpoint, set that up

```
def train_model(config: ModelConfig):
    # Define the device
    assert torch.cuda.is_available(), "Training on CPU is not supported"
    device = torch.device("cuda")
    print(f"GPU {config.local_rank} - Using device: {device}")

    # Make sure the weights folder exists
    Path(config.model_folder).mkdir(parents=True, exist_ok=True)

    # Load the dataset
    print(f"GPU {config.local_rank} - Loading dataset...")
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, eps=1e-9)

    # By default, load the latest checkpoint
    initial_epoch = 0
    global_step = 0
    wandb_run_id = None
    if config.preload != '':

        if config.preload == 'latest':
            # Get the filename of the latest checkpoint
            model_filename = get_latest_weights_file_path(config)
        else:
            # In case we want to preload a specific checkpoint
            model_filename = get_weights_file_path(config, int(config.preload))

        if model_filename is not None:
            print(f'GPU {config.local_rank} - Preloading model {model_filename}')
            state = torch.load(model_filename)
            model.load_state_dict(state['model_state_dict'])
            initial_epoch = state['epoch'] + 1
            optimizer.load_state_dict(state['optimizer_state_dict'])
            global_step = state['global_step']
            wandb_run_id = state['wandb_run_id']
            del state
        else:
            # If we couldn't find a model to preload, just start from scratch
            print(f'GPU {config.local_rank} - Could not find model to preload: {config.preload}. Starting from scratch')

    # Only initialize W&B on the global rank 0 node
    if config.local_rank == 0:
        wandb.init(
            # set the wandb project where this run will be logged
            project="pytorch-transformer-distributed",
            # allow resuming existing run with the same name (in case the rank 0 node crashed)
            name=f"global_rank_{config.global_rank}",
            id=wandb_run_id,
            resume="allow",
            group=config.wandb_group,
            # track hyperparameters and run metadata
            config=config
        )

    # Convert the model to DistributedDataParallel
    # Here we can also specify the bucket_cap_mb parameter to control the size of the buckets
    model = DistributedDataParallel(model, device_ids=[config.local_rank])

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    if config.global_rank == 0:
        # define our custom x axis metric
        wandb.define_metric("global_step")
        # define which metrics will be plotted against it
        wandb.define_metric("validation/*", step_metric="global_step")
        wandb.define_metric("train/*", step_metric="global_step")

    for epoch in range(initial_epoch, config.num_epochs):
        torch.cuda.empty_cache()
        model.train()

        # Disable tqdm on all nodes except the rank 0 GPU on each server
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d} on rank {config.global_rank}", disable=config.local_rank != 0)

        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device) # (b, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (B, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (B, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (B, 1, seq_len, seq_len)

            # # Run the tensors through the encoder, decoder and the projection layer
            # encoder_output = model.module.encode(encoder_input, encoder_mask) # (B, seq_len, d_model)
            # decoder_output = model.module.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (B, seq_len, d_model)
            # proj_output = model.module.project(decoder_output) # (B, seq_len, vocab_size)
            proj_output = model(encoder_input, encoder_mask, decoder_input, decoder_mask)

            # Compare the output with the label
            label = batch['label'].to(device) # (B, seq_len)

            # Compute the loss using a simple cross entropy
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}", "global_step": global_step})

            if config.local_rank == 0:
                # Log the loss
                wandb.log({'train/loss': loss.item(), 'global_step': global_step})

            # Backpropagate the loss
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

        # Only run validation and checkpoint saving on the rank 0 node
        if config.global_rank == 0:
            # Run validation at the end of every epoch
            run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config.seq_len, device, lambda msg: batch_iterator.write(msg), global_step)

            # Save the model at the end of every epoch
            model_filename = get_weights_file_path(config, epoch)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(), # Need to access module because we are using DDP
                'optimizer_state_dict': optimizer.state_dict(),
                'global_step': global_step,
                'wandb_run_id': wandb.run.id # Save to resume logging data
            }, model_filename)
```

## When does pytroch synchronize gradients?

- everytime you call loss.backward
- each node calculates local graidents (derivative of loss function wrt each node of the computational graph)
- each node will send local gradient to one single node and received back cumulative gradient
- each node will update its weights using cumulative gradient and local optimizer

use no_sync() context to let gradient acculuate without synchronizing

## Computation computation overlap

- gpu performas forward backward, then idle due to communication overhead (all reduce), pytroch communicates gradient of a node as soon as its available.
- create buckets of gradients
- while computing backward step, you can communicate the graidents and receive back the cumulative

- everytime a bucket is available you send the bucket and received back the cumulative.
- 25 MB as the size of the bucket is recommended
