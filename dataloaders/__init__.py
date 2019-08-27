from dataloaders.datasets import icdar
#from dataloaders.datasets import pascal
from dataloaders.datasets import totalText

from torch.utils.data import DataLoader

def make_data_loader(args, **kwargs):

    if args.dataset == 'icdar':
        train_set = icdar.ICDARSegmentation(args, split='train')
        val_set = icdar.ICDARSegmentation(args, split='val')
        # if args.use_sbd:
        #     sbd_train = sbd.SBDSegmentation(args, split=['train', 'val'])
        #     train_set = combine_dbs.CombineDBs([train_set, sbd_train], excluded=[val_set])

        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = None

        return train_loader, val_loader, test_loader, num_class

    if args.dataset == 'totalText':
        train_set = totalText.TotalTextSegmentation(args, split='train')
        val_set = totalText.TotalTextSegmentation(args, split='val')
        # if args.use_sbd:
        #     sbd_train = sbd.SBDSegmentation(args, split=['train', 'val'])
        #     train_set = combine_dbs.CombineDBs([train_set, sbd_train], excluded=[val_set])

        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = None

        return train_loader, val_loader, test_loader, num_class


    # if args.dataset == 'pascal':
    #     train_set = pascal.VOCSegmentation(args, split='train')
    #     val_set = pascal.VOCSegmentation(args, split='val')
    #     if args.use_sbd:
    #         sbd_train = sbd.SBDSegmentation(args, split=['train', 'val'])
    #         train_set = combine_dbs.CombineDBs([train_set, sbd_train], excluded=[val_set])

    #     num_class = train_set.NUM_CLASSES
    #     train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
    #     val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
    #     test_loader = None

    #     return train_loader, val_loader, test_loader, num_class

    else:
        raise NotImplementedError

