class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'icdar_256_resized':
            return '/path/to/datasets/TextSegmentation/ICDAR2013_KAIST/resized_256/'
        elif dataset == 'totalText':
            return '/path/to/datasets/TextSegmentation/total_text/'
        elif dataset == 'icdar':
            return '/path/to/datasets/TextSegmentation/ICDAR2013_KAIST/'
        elif dataset == 'icdar2015':
            return 'data/'            
        if dataset == 'pascal':
            return '/path/to/dataset/pascalVOC2012/' 
        elif dataset == 'sbd':
            return '/path/to/datasets/benchmark_RELEASE/'  
        elif dataset == 'cityscapes':
            return '/path/to/dataset/cityscapes/' 
        elif dataset == 'coco':
            return '/path/to/datasets/coco/'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
