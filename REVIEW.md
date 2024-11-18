# MagMax

    --model ${model} \
    --dataset ${dataset} \
    --epochs ${epochs} \
    --n_splits ${n_splits} \
    --split_strategy class \
    --sequential-finetuning \
    --seed ${seed} \
        |& tee ${out_dir}/splits:${n_splits}-ep:${epochs}-seed:${seed}.out

1. [finetune tasks using](#finetune_splittedpy), save [ImageClassifier](#imageclassifier) [encoder](#imageencoder)'s task vector ($\theta_i$) to disk

2. [merge task](#merge_splittedpy)

## Dataset

## Model

### [ImageClassifier](src/modeling.py#L99)
>
> base model contains `self.image_encoder` and `self.classification_head`
>

- `self.image_encoder`: [ImageEncoder](#imageencoder)

- `self.classification_head`: [ClassificationHead](#classificationhead)

### [ImageEncoder](src/modeling.py#L7)

### [ClassificationHead](src/modeling.py#L56)

- Is this used in all tasks?

## [TaskVector](src/merging/task_vectors.py#L4)

- By default, `finetuned_state_dict` is `finetuned_checkpoint`'s state_dict
- `self.vector`: a dict of parameters whose keys are `pretrained_state_dict`'s keys. It values are differences between `finetuned_state_dict` and `pretrained_state_dict`

- `sefl.apply_to`: apply self params into pretrained state dict which is params of [ImageEncoder](#imageencoder)

## finetune_splitted.py

load [ImageClassifier](#imageclassifier)

## merge_splitted.py

- init list of [TaskVector](#taskvector), then [search_evaluate_merging](merge_splitted.py#L62) by 3 hypothesis: random selection, magnitude max selection, and average

- apply merge strategy into task vectors to get desired task vector $\tau_{MagMax}$
- eval the found $\tau_{MagMax}$ on data
