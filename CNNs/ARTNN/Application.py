import torch
import tkinter as tk
from PIL import ImageTk, Image
import os
from tkinter import filedialog
from tkinter import messagebox
from functools import partial
import Art_nn

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.user_im_dir = None
        self.artist_im_dir = None
        self.output_image = None
        self.images = []
        self.images_panels = []
        self.images_buttons = []
        self.images_frames = []
        self.pack()
        self.create_main_widget()
        self.__load_images(200)

    def select_artist(self, pathfile):
        self.artist_im_dir = pathfile
        self.selected_image_label["text"] = pathfile

    def select_image(self):
        currdir = os.path.dirname(os.path.realpath(__file__))
        self.user_im_dir = filedialog.askopenfile(parent=self.select_path_frame, initialdir=currdir,
                                             title="select your image").name

        dim = 200
        img = Image.open(self.user_im_dir)
        img = img.resize((dim, dim), Image.ANTIALIAS)

        self.im_label.configure(image=ImageTk.PhotoImage(img))
        self.user_image_text["text"] = self.user_im_dir
        return

    def run(self):
        try:
            assert self.user_im_dir != None
            assert self.artist_im_dir != None
        except AssertionError:
            messagebox.showerror("Error!", "you must select an artist and a personal picture")
            return

        device = "cuda" if torch.cuda.is_available() else "cpu"

        art = Art_nn.ArtNN()
        print(art)
        art.load_images(self.artist_im_dir, self.user_im_dir)
        self.output_image = art.run()

        answer = messagebox.askyesno("save the image","do you want to save the output image?")

        if answer is True:
            currdir = os.path.dirname(os.path.realpath(__file__))
            path = filedialog.askdirectory(parent=self.select_path_frame, initialdir=currdir,
                                             title="select output folder")
            print(path)

            try:
                assert path != None
            except AssertionError:
                messagebox.showerror("Error!", "there's an error with the path, please contact us!")

            try:
                art.save_image(path, self.output_image)
                messagebox.showinfo(title="Save done!", message="Image saved successfully!")
            except Exception:
                messagebox.showerror(title="Error", message="There is an error in saving the image, please contact us")

    def create_main_widget(self):
        frame = tk.Frame(self.master)
        frame.pack(side="top", fill="y", expand="yes")

        #mostra quale artista ha selezionato
        self.selected_image_frame = tk.Frame(frame)
        self.selected_image_frame.pack(side="top", fill="x", expand="false")

        self.selected_image_text = tk.Label(self.selected_image_frame, text="selected the artist: ")
        self.selected_image_text.pack(side="left", fill="y", expand="false")

        self.selected_image_label = tk.Label(self.selected_image_frame, text="no image")
        self.selected_image_label.pack(side="left", fill="y", expand="false")

        #selezione dell'immagine da parte dell'utente
        self.select_path_frame = tk.Frame(frame)
        self.select_path_frame.pack(side="top", fill="x", expand="false")

        self.select_image_text = tk.Label(self.select_path_frame, text="selected your image: ")
        self.select_image_text.pack(side="left", fill="y", expand="false")

        self.select_dir_button = tk.Button(self.select_path_frame, text="select", command=self.select_image)
        self.select_dir_button.pack(side="left", fill="y", expand="false")

        #mostra all'utente l'immagine scelta
        self.user_image_frame = tk.Frame(frame)
        self.user_image_frame.pack(side="top", fill="x", expand="false")

        self.user_image_label = tk.Label(self.user_image_frame, text="your image: ")
        self.user_image_label.pack(side="left", fill="y", expand="false")

        self.user_image_text = tk.Label(self.user_image_frame, text="not selected")
        self.user_image_text.pack(side="left", fill="y", expand="false")

        #bottone per avviare la baracca
        self.run_frame = tk.Frame(frame)
        self.run_frame.pack(side="top", fill="x", expand="false")

        self.run_button = tk.Button(self.run_frame, text="run", command=self.run)
        self.run_button.pack(side="top", fill="y", expand="false")

        #immagine di output
        self.im_frame = tk.Frame(frame)
        self.im_frame.pack(side="top", fill="x", expand="false")

        self.im_label = tk.Label(self.im_frame,image=None)
        self.im_label.pack(side="left", fill="both", expand="yes")

    def __load_images(self, dim):
        for _, _, files in os.walk(r"{}/Artists".format(os.path.dirname(os.path.realpath(__file__)))):
            for fname in files:
                img = Image.open(r"{}/Artists/{}".format(os.path.dirname(os.path.realpath(__file__)), fname))
                img = img.resize((dim, dim), Image.ANTIALIAS)
                self.images.append([ImageTk.PhotoImage(img), fname])

        for image, fname in self.images:
            image_frame = tk.Frame(self.master)
            image_label = tk.Label(image_frame, image=image)

            button = tk.Button(image_frame, text=fname[:-4], command=partial(self.select_artist, r"{}/Artists/{}".format(os.path.dirname(os.path.realpath(__file__)), fname)))

            image_label.pack(side="top",fill="y", expand="no")
            button.pack(side="top", fill="y", expand="false")
            image_frame.pack(side="left", fill="both", expand="yes")

            self.images_buttons.append(button)
            self.images_frames.append(image_frame)
            self.images_panels.append(image_label)

        self.master.mainloop()