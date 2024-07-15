import os
os.environ['OPENBLAS_NUM_THREADS'] = '64'
import torch
import torchvision.datasets as datasets
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import open_clip
from tqdm import tqdm
from my_training.data import MCLDataset 
import argparse

def main():
    parser = argparse.ArgumentParser(description='Clustering for next stage training')
    parser.add_argument('--device', type=str, default='cuda:0', help='CUDA device')
    parser.add_argument('--stage', type=int, default=0, help='Stage of MCL, 0 for init vanilla model')
    parser.add_argument('--modelarch', type=str, default='RN50-quickgelu', help='Model architecture of open_clip')
    parser.add_argument('--pretrained', type=str, default='openai', help='Pretrained model version of open_clip')
    parser.add_argument('--checkpoint-dir', type=str, default=None, help='Checkpoint directory of the last stage model')
    parser.add_argument('--img-file-path', type=str, default=None, help='Image file path of the training dataset')
    parser.add_argument('--save-path', type=str, default='./save', help='Path to save the outputs')
    parser.add_argument('--num-clusters', type=int, default=10, help='Number of clusters of the K-means')
    parser.add_argument('--cluster-batch-size', type=int, default=1000000, help='batch_size of the K-means')
    parser.add_argument('--cluster-max-iter', type=int, default=100, help='max_iter of the K-means')
    parser.add_argument('--cluster-tol', type=float, default=1e-5, help='tol of the K-means')
    parser.add_argument('--inference-batch-size', type=int, default=256*16, help='batch_size of the inference')
    parser.add_argument('--num-workers', type=int, default=64, help='num_workers of the inference')
    args = parser.parse_args()

    device = torch.device(args.device)
    stage = args.stage
    modelarch = args.modelarch
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(modelarch, pretrained=args.pretrained,force_quick_gelu=True)
    checkpoint_dir=args.checkpoint_dir
    savepath=args.save_path
    num_clusters = args.num_clusters


    if checkpoint_dir is not None:
        checkpoint =torch.load(checkpoint_dir, map_location='cpu')
        sd = checkpoint["state_dict"]
        if next(iter(sd.items()))[0].startswith('module'):
            sd = {k[len('module.'):]: v for k, v in sd.items()}
        model.load_state_dict(sd)
    model = model.to(device)
    #inference to obtain representations of all the training dataset
    my_dataset = MCLDataset(args.img_file_path,transform=preprocess_val,tokenizer=None)
    my_dataloader = DataLoader(my_dataset, batch_size=args.inference_batch_size, shuffle=False,drop_last=False,num_workers=args.num_workers,pin_memory=True)
    image_features = []
    print('inferencing:')
    with torch.no_grad(), torch.cuda.amp.autocast():
        for img, _ in tqdm(my_dataloader):
            img = img.to(device)
            image_feature = model.encode_image(img)
            image_feature /= image_feature.norm(dim=-1, keepdim=True)
            image_features.append(image_feature.cpu())
    image_features = torch.cat(image_features,dim=0)
    features=image_features.numpy()

    #do clustering and save
    from sklearn.cluster import MiniBatchKMeans
    kmeans = MiniBatchKMeans(
        n_clusters=num_clusters, 
        batch_size=args.cluster_batch_size, 
        max_iter=args.cluster_max_iter,
        tol=args.cluster_tol,
        verbose=1)

    kmeans.fit(features)

    centroids=kmeans.cluster_centers_
    torch.save(torch.Tensor(centroids),os.path.join(savepath, '{}_{}_{}.pt'.format(modelarch, stage, 'centroids')))
    cluster_labels = torch.LongTensor(kmeans.predict(features))
    torch.save(cluster_labels,os.path.join(savepath, '{}_{}_{}.pt'.format(modelarch, stage, 'cluster_labels')))

    #integrate the clustering labels from the previous stages to get the pseudo labels
    cls=torch.zeros(cluster_labels.shape[0],dtype=torch.long)
    for s in range(stage+1):
        cl=torch.load(os.path.join(savepath, '{}_{}_{}.pt'.format(modelarch, stage, 'cluster_labels')))
        cls=cls*num_clusters+cl
    torch.save(cls,os.path.join(savepath, '{}_{}_{}.pt'.format(modelarch, stage, 'pseudo_labels')))

if __name__ == '__main__':
    main()