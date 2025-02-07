import torch
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from lightning_module import GAN

class GANImageGenerator:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()
        self.latent_dim = 100

    def generate(self):
        """Generate single image from random noise."""
        noise = torch.randn(1, self.latent_dim, device=self.device)
        generated_image = self.model(noise)
        
        generated_image = generated_image.squeeze(0)
        if generated_image.shape[0] == 1:
            generated_image = generated_image.squeeze(0)
        elif generated_image.shape[0] == 3:
            generated_image = generated_image.permute(1, 2, 0)

        generated_image = (generated_image.detach().cpu().numpy() * 255).astype('uint8')
        return Image.fromarray(generated_image).resize((256, 256))

    @staticmethod
    def convert_checkpoint_to_torchscript():
        """Convert Lightning checkpoint to TorchScript."""
        try:
            checkpoint_path = filedialog.askopenfilename(title="Select PyTorch Lightning checkpoint")
            if not checkpoint_path:
                return None
            
            model = GAN.load_from_checkpoint(checkpoint_path)
            model.eval()
            
            script_path = "model.pt"
            traced_model = model.to_torchscript(file_path=script_path, method="script")
            messagebox.showinfo("Conversion", f"Model converted and saved to {script_path}")
            return script_path
        except Exception as e:
            messagebox.showerror("Conversion Error", str(e))
            return None

class GANImageApp:
    def __init__(self, generator):
        self.generator = generator
        self.root = tk.Tk()
        self.root.title("GAN Image Generator")
        
        self.canvas = tk.Canvas(self.root, width=256, height=256)
        self.canvas.pack()
        
        self.current_image = None
        
        tk.Button(self.root, text="Generate Image", command=self.render_image).pack()
        tk.Button(self.root, text="Save Image", command=self.save_image).pack()
        tk.Button(self.root, text="Convert Checkpoint", command=self.convert_checkpoint).pack()
        
    def render_image(self):
        image = self.generator.generate()
        img_tk = ImageTk.PhotoImage(image=image)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=img_tk)
        self.canvas.image = img_tk
        self.current_image = image
    
    def save_image(self):
        if self.current_image:
            path = filedialog.asksaveasfilename(defaultextension=".png")
            if path:
                self.current_image.save(path)
                messagebox.showinfo("Save", f"Image saved to {path}")
    
    def convert_checkpoint(self):
        script_path = GANImageGenerator.convert_checkpoint_to_torchscript()
        if script_path:
            # Optionally reload the model after conversion
            self.generator = GANImageGenerator(script_path)
    
    def run(self):
        self.root.mainloop()

def main():
    generator = GANImageGenerator("model.pt")
    app = GANImageApp(generator)
    app.run()

if __name__ == "__main__":
    main()