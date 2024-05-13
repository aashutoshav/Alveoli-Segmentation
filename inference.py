import torch
import os
from torchvision import transforms
from PIL import Image
from run import UNetR

def save_image(output, output_path):
    image = transforms.ToPILImage()(output)
    image.save(output_path)


def generate_images(model_state_dict, input_images_folder, output_folder):
    input_images = []
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    for filename in os.listdir(input_images_folder):
        image_path = os.path.join(input_images_folder, filename)
        image = Image.open(image_path)
        image = transform(image)
        input_images.append(
            image.unsqueeze(0)
        )

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    model = UNetR(
        {
            "n_channels": 3,
            "patch_size": 16,
            "embed_dim": 768,
            "num_patches": 256,
            "num_layers": 12,
            "num_heads": 8,
            "mlp_dim": 2048,
            "dropout_rate": 0.1,
            "img_size": 256,
        }
    )
    model.load_state_dict(model_state_dict)
    model.eval()
    with torch.inference_mode():
        for idx, input_image in enumerate(input_images):
            output_path = os.path.join(output_folder, f"output{idx}.jpg")
            outputs = model(input_image)
            save_image(
                outputs[0], output_path
            )


if __name__ == "__main__":
    model_state_dict = torch.load("model.pt")
    input_images_folder = "images"
    output_folder = "outputs"
    generate_images(model_state_dict, input_images_folder, output_folder)
