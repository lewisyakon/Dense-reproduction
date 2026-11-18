# Storage layout

This project separates source code, active training files, and archival copies.

## Server-local paths

- Repository: `/home/lewis/DenseLatentCommunication`
- Python environment: `/mnt/hd/lewis/conda-envs/c2c`
- Model weights: `/mnt/hd/lewis/models/Qwen3-4B` and
  `/mnt/hd/lewis/models/Qwen3-8B`
- Active data: `/mnt/hd/lewis/DenseLatentCommunication/data`
- Active artifacts: `/mnt/hd/lewis/DenseLatentCommunication/artifacts`
- Formal direction outputs: `/mnt/hd/lewis/DenseLatentCommunication/4B_to_8B`
  and `/mnt/hd/lewis/DenseLatentCommunication/8B_to_4B`

Only files used frequently during GPU training should stay on `/mnt/hd`.
The raw Hugging Face cache path `/mnt/hd/lewis/huggingface` is a compatibility
symlink to `/mnt/nfs/cloud/cache/huggingface`, so cache downloads do not
consume scarce `/mnt/hd` space. The Qwen3-4B and Qwen3-8B model directories
above remain server-local because formal training loads them repeatedly.

The repository's ignored `artifacts` path is a compatibility symlink to the
active artifacts directory on `/mnt/hd`.

## NFS archival paths

- Repository copy: `/mnt/nfs/cloud/DenseLatentCommunication`
- Model archive: `/mnt/nfs/cloud/models/DenseLatentCommunication`
- Data archive: `/mnt/nfs/cloud/data/DenseLatentCommunication`
- Checkpoint archive: `/mnt/nfs/cloud/checkpoints/DenseLatentCommunication`
- Hugging Face cache: `/mnt/nfs/cloud/cache/huggingface`

Do not run training with the NFS archive as the output directory. Copy active
inputs to `/mnt/hd` before training, write checkpoints locally, and sync
completed outputs to NFS afterward.

## Synchronization

Sync code without large ignored files:

```bash
rsync -a --exclude=/artifacts --exclude=/data \
  /home/lewis/DenseLatentCommunication/ \
  /mnt/nfs/cloud/DenseLatentCommunication/
```

Archive active data and checkpoints:

```bash
rsync -a /mnt/hd/lewis/DenseLatentCommunication/data/ \
  /mnt/nfs/cloud/data/DenseLatentCommunication/

rsync -a /mnt/hd/lewis/DenseLatentCommunication/artifacts/ \
  /mnt/nfs/cloud/checkpoints/DenseLatentCommunication/
```

Restore archived data and checkpoints:

```bash
rsync -a /mnt/nfs/cloud/data/DenseLatentCommunication/ \
  /mnt/hd/lewis/DenseLatentCommunication/data/

rsync -a /mnt/nfs/cloud/checkpoints/DenseLatentCommunication/ \
  /mnt/hd/lewis/DenseLatentCommunication/artifacts/
```

Before any NFS synchronization, verify that the mount is active:

```bash
findmnt -T /mnt/nfs/cloud
```

The source must be an NFS filesystem, not the server root filesystem.
