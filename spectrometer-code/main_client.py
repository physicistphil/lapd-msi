from GUI_client import SpecApp  # SaveData
import tkinter as tk

if __name__ == '__main__':
    root = tk.Tk()
    myapp = SpecApp(root)
    root.geometry("1024x640")
    root.title("Spectrometer GUI")
    root.config(bg='#345')
    myapp.mainloop()
