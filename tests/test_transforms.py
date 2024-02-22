import torch
from torchvision import transforms
from ser.transforms import flip, normalize, _configure_transforms
from numpy import mean, std

def test_flip():
    x = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    x = torch.tensor(x)
    flipped_tensor = flip()(x)
    assert torch.all(flipped_tensor == torch.tensor(
        [[9, 8, 7],
        [6, 5, 4],
        [3, 2, 1]])
    ) == True

def test_normalize():
    x = [[[1, 2, 3], [1, 2, 3], [1, 2, 3]]]
    x = torch.tensor(x, dtype=torch.float)
    normalized_tensor = normalize()(x)
    assert normalized_tensor == torch.tensor([[[1, 3, 5],
                                        [1, 3, 5],
                                        [1, 3, 5]]],
                                        dtype=torch.float)
    
    
def test_configure_transforms():
    x = [[[1, 2, 3], [1, 2, 3], [1, 2, 3]]]
    x = torch.tensor(x, dtype=torch.float32)
    transform = _configure_transforms(flip_img=True)
    transformed_x = transform(x)
    assert transformed_x == torch.tensor([[[5, 3, 1],
                                    [5, 3, 1],
                                    [5, 3, 1]]],
                                    dtype=torch.float32)
    

