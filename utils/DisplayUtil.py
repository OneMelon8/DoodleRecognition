'''
    Utility class for displaying images visually
'''
# Imports
import math
import numpy as np
import tkinter as tk
import utils.FileUtil as fUtil

# Constants
SIZE_SCALE = 20.0;
SIZE_SCALE_BULK = 2.5;


# Methods
def show_image(np_arr, bulk_size=36, name="Data Preview"):
    master = tk.Tk()
    master.title(name)
    
    if len(np_arr) == 784:
        # Single image
        canvas = tk.Canvas(master, width=28 * SIZE_SCALE, height=28 * SIZE_SCALE)
        canvas.pack()
    
        for a in range(28):
            for b in range(28):
                # Check if need to draw
                if np_arr[a * 28 + b] == 0:
                    continue
                # Draw
                x, y = b * SIZE_SCALE, a * SIZE_SCALE
                canvas.create_rectangle(x, y, x + SIZE_SCALE, y + SIZE_SCALE, fill="#000000")
    elif type(np_arr[0]) == np.ndarray:
        # Bulk display images
        limit = math.sqrt(bulk_size)
        canvas = tk.Canvas(master, width=28 * SIZE_SCALE_BULK * limit, height=28 * SIZE_SCALE_BULK * limit)
        canvas.pack()
        
        for i in range(bulk_size):
            image_data = np_arr[i]

            def draw_image(starting_x, starting_y):
                for a in range(28):
                    for b in range(28):
                        # Check if need to draw
                        if image_data[a * 28 + b] == 0:
                            continue
                        # Draw
                        x, y = starting_x + b * SIZE_SCALE_BULK, starting_y + a * SIZE_SCALE_BULK
                        canvas.create_rectangle(x, y, x + SIZE_SCALE_BULK, y + SIZE_SCALE_BULK, fill="#000000")
            
            draw_image(i % limit * 28 * SIZE_SCALE_BULK, i // limit * 28 * SIZE_SCALE_BULK)

    else:
        raise ValueError("Invalid numpy array: requires 1D array of an image or 2D array of a bulk of images")
    
    master.mainloop()


def show_drawable_canvas():
    thickness = 10
    draw_data = [[0] * 28 for _ in range(28)]

    def draw(event):
        nonlocal draw_data
        canvas.create_oval(event.x - thickness, event.y - thickness, event.x + thickness, event.y + thickness, fill="#000000")
        if event.x // 20 < 28 and event.y // 20 < 28:
            draw_data[event.y // 20][event.x // 20] = 1

    def clear(event):
        nonlocal draw_data
        event.widget.delete("all")
        draw_data = [[0] * 28 for _ in range(28)]
        
    def done(_):
        master.destroy()
    
    master = tk.Tk()
    canvas = tk.Canvas(master, width=560, height=560)
    canvas.pack()
    canvas.bind('<ButtonPress-1>', draw)
    canvas.bind('<B1-Motion>', draw)
    canvas.bind('<Double-1>', clear)
    canvas.bind('<ButtonPress-2>', done)
    
    master.mainloop()
    return np.array([a for b in draw_data for a in b])

    
if __name__ == "__main__":
    dataset_index = -3
    data = fUtil.load_data(fUtil.TRAINING_DATA_NAMES[dataset_index])
    name = fUtil.TRAINING_DATA_NAMES[dataset_index][:1].upper() + fUtil.TRAINING_DATA_NAMES[dataset_index][1:-4]
    show_image(data[0])
    show_image(data, bulk_size=49, name=name)
    data = show_drawable_canvas()
    print(data)
    show_image(data)
