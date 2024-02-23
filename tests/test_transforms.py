import torch
from torchvision import transforms
from ser.transforms import flip, normalize, configure_transforms
from numpy import mean, std

def test_flip():
    x = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    x = torch.tensor(x)
    flipped_tensor = flip()(x)
    expected = torch.tensor(
                            [[9, 8, 7],
                            [6, 5, 4],
                            [3, 2, 1]])
    assert torch.all(flipped_tensor == expected)

def test_normalize():
    x = [[[1, 2, 3], [1, 2, 3], [1, 2, 3]]]
    x = torch.tensor(x, dtype=torch.float)
    normalized_tensor = normalize()(x)
    expected = torch.tensor([[[1, 3, 5],
                            [1, 3, 5],
                            [1, 3, 5]]],
                            dtype=torch.float)
    assert torch.all(normalized_tensor == expected)
    
    
def test_configure_transforms():
    x = [[[1, 2, 3], [1, 2, 3], [1, 2, 3]]]
    x = torch.tensor(x, dtype=torch.float32)
    transform = configure_transforms(flip_img=True)
    assert len(transform) == 2
    transform_2 = configure_transforms(flip_img=False)
    assert len(transform_2) == 1
 
    # transform = transforms.Compose(transform)
    # transformed_x = transform()(x)
    # expected = torch.tensor([[[5, 3, 1],
    #                         [5, 3, 1],
    #                         [5, 3, 1]]],
    #                         dtype=torch.float32)
    # assert torch.all(transformed_x == expected)



if __name__ == "__main__":
    test_configure_transforms()
