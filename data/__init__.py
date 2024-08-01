from torch.utils.data import DataLoader, BatchSampler, SequentialSampler

def prepare_dataloader(dataset, batch_size, num_shots, sampler=None, instruction=None):
    if sampler is None:
        sampler = SequentialSampler(dataset)

    def collate_fn(batch):
        batch_images, batch_context, batch_answer = [], [], []
        for i in range(0, len(batch), num_shots + 1):
            mini_batch = batch[i : i + num_shots + 1]
            batch_images.append([data["image"] for data in mini_batch])
            batch_answer.append(mini_batch[-1]["answer"])
            messages = []
            if instruction is not None:
                messages.append({"role": "instruction", "content": instruction})

            for data in mini_batch[:-1]:
                messages.append(
                    {
                        "role": "example",
                        "query": [
                            {"type": "image"},
                            {"type": "text", "text": data["question"]},
                        ],
                        "answer": [
                            {
                                "type": "text",
                                "text": data["answer"],
                            },
                        ],
                    }
                )
            messages.append(
                {
                    "role": "question",
                    "query": [
                        {"type": "image"},
                        {"type": "text", "text": mini_batch[-1]["question"]},
                    ],
                }
            )
            batch_context.append(messages)
        
        return batch_images, batch_context, batch_answer

    return DataLoader(
        dataset,
        collate_fn=collate_fn,
        batch_sampler=BatchSampler(
            sampler, batch_size * (num_shots + 1), drop_last=True
        ),
    )
