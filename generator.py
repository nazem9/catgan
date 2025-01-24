import torch
from tkinter import Tk, Label, Button, filedialog, Canvas

from PIL import Image, ImageTk
from lightning_module import GAN

def make_torchscript():
    """Create a Torchscript file from a Pytorch lightning checkpoint."""
    checkpoint_path = filedialog.askopenfilename(title="Select a PyTorch Lightning checkpoint file") 
    print(checkpoint_path)
    # Load the model from the checkpoint
    model = GAN.load_from_checkpoint(checkpoint_path)
    model.eval()

    # Convert the model to Torchscript
    traced_model = model.to_torchscript(file_path="model.pt", method="script")


# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "model.pt"
model = torch.jit.load(model_path, map_location=device)
model.eval()

def generate_image():
    """
    Generate an image using the model and display it in the Tkinter app.
    """
    try:
        # Generate random noise as input
        latent_dim = 100  # Replace with your model's latent dimension
        noise = torch.randn(1, latent_dim, device=device)  # Ensure noise is on the same device as the model

        # Forward pass
        generated_image = model(noise)

        # Check the shape of the output tensor
        print(f"Generated image tensor shape: {generated_image.shape}")

        # Handle single-channel or multi-channel images
        generated_image = generated_image.squeeze(0)  # Remove batch dimension
        if generated_image.shape[0] == 1:
            # Grayscale image: Remove channel dimension
            generated_image = generated_image.squeeze(0)
        elif generated_image.shape[0] == 3:
            # RGB image: Transpose to (H, W, C)
            generated_image = generated_image.permute(1, 2, 0)

        # Convert to NumPy and scale to [0, 255]
        generated_image = (generated_image.detach().cpu().numpy() * 255).astype('uint8')

        # Create a PIL image
        image = Image.fromarray(generated_image)
        image = image.resize((256, 256))

        # Display the image in the Tkinter canvas
        img_tk = ImageTk.PhotoImage(image=image)
        canvas.image = img_tk  # Keep a reference to avoid garbage collection
        canvas.create_image(0, 0, anchor="nw", image=img_tk)
    except Exception as e:
        print(f"Error generating image: {e}")


# Create the Tkinter app
root = Tk()
root.title("GAN Image Generator")

# Add a canvas to display the generated image
canvas = Canvas(root, width=256, height=256)  # Adjust dimensions as per your model output
canvas.pack()

# Add a button to trigger image generation
generate_button = Button(root, text="Generate Image", command=generate_image)
generate_button.pack()

convert_button = Button(root, text="convert to torchscript", command = lambda : make_torchscript())
convert_button.pack()
# Start the Tkinter event loop
root.mainloop()
