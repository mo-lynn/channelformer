from data_provider.data_loader import UEAloader, load_uea_dataset
from data_provider.uea import collate_fn
from torch.utils.data import DataLoader

data_dict = {
    'UEA': UEAloader
}

def data_provider(args, data, labels, class_names, max_sqe_len, shuffle_flag):
    Data = data_dict[args.dataset]

    batch_size = args.batch_size

    drop_last = False
    data_set = Data(
        args,
        data,
        labels,
        class_names,
        max_sqe_len,
    )

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
        collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
    )
    return data_set, data_loader


def load_dataset(args, root_path):
    if args.dataset == 'UEA':
        dataset = load_uea_dataset(args, root_path)  # Load UEA *.ts Data
    return dataset