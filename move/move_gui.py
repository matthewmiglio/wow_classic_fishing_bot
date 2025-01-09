import tkinter as tk
from tkinter import ttk
from move.move_functions import (
    ratchet_inn_chair_to_wc_fish_spot,
    gadgetzan_inn_to_tanaris_shore,
)


def open_move_gui():
    root = tk.Tk()
    root.title("Simple GUI")

    label = tk.Label(root, text="Select an option and click a button:")
    label.pack(pady=10)

    options = ["Gedgetzan inn to Tanaris shore", "Ratchet inn chair to WC entrance"]
    dropdown = ttk.Combobox(root, values=options)
    dropdown.pack(pady=10)

    def green_button_pressed():
        selected_option = dropdown.get()
        print(f"{selected_option} - Green button pressed")

        if selected_option == "Gedgetzan inn to Tanaris shore":
            print("Running to tanaris fish spot...")
            gadgetzan_inn_to_tanaris_shore()
            print("Done running to Tanaris shore spot")
        elif selected_option == "Ratchet inn chair to WC entrance":
            print("Running from Ratchet inn chair all the way to the WC fish spot!")
            ratchet_inn_chair_to_wc_fish_spot()
            print("Done running to WC fish spot")

    def red_button_pressed():
        print("Close button clicked")
        root.quit()

    green_button = tk.Button(root, text="Run", bg="green", command=green_button_pressed)
    green_button.pack(pady=10)

    red_button = tk.Button(root, text="Close", bg="red", command=red_button_pressed)
    red_button.pack(pady=10)

    root.mainloop()
    root.destroy()


if __name__ == "__main__":
    open_move_gui()
