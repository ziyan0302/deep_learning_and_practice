import os
import torch
import torch.nn as nn
import numpy as np
import copy
from PIL import Image
from util import get_test_conditions,save_image
from evaluator import EvaluationModel
from torchvision import transforms
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test(dataloader, g_model, d_model, z_dim):
    test_conditions = get_test_conditions(os.path.join('dataset/task_1', 'test.json')).to(device)
    print(f'test_conditions: {test_conditions.shape}')
    new_test_conditions = get_test_conditions(os.path.join('dataset', 'new_test.json')).to(device)
    ground_test_conditions = get_test_conditions(os.path.join('dataset/task_1', 'ground.json')).to(device)
    print(f'ground_test_conditions: {ground_test_conditions.shape}')
    fixed_z = random_z(len(test_conditions), z_dim).to(device)
    evaluation_model = EvaluationModel()
    ground_truth = Image.open(os.path.join('dataset/task_1/images', 'CLEVR_train_006003_2.png')).convert('RGB')
    transformations = transforms.Compose([transforms.Resize((64,64)), transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    ground_truth = transformations(ground_truth)
    ground_truth = torch.unsqueeze(ground_truth, 0)
    ground_truth = ground_truth.to(device)
    print(f'ground_tr.shape: {ground_truth.shape}')

    g_model.eval()
    d_model.eval()
    with torch.no_grad():
        gen_imgs = g_model(fixed_z, test_conditions)
        new_gen_imgs = g_model(fixed_z, new_test_conditions)
        print(f'gen_imgs.shape: {gen_imgs.shape}')
    ground_score = evaluation_model.eval(ground_truth, ground_test_conditions)
    score = evaluation_model.eval(gen_imgs, test_conditions)
    new_score = evaluation_model.eval(new_gen_imgs, new_test_conditions)

    print(f'testing score: {score:.2f}')
    print(f'new_testing score: {new_score:.2f}')
    print(f'ground score: {ground_score:.2f}')

def random_z(batch_size, z_dim):
    return torch.randn(batch_size, z_dim)