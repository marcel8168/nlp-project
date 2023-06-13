import tkinter


class GUI:
    def __init__(self):
        pass

    def show_custom_popup(self, titel: str, message: str) -> None:
        """
        Displays a custom popup window with the given title and message.

        Arguments
        ---------
            title (str): The title of the popup window.
            message (str): The message to be displayed in the popup window.
        """
        popup = tkinter.Tk()
        popup.title(titel)
        popup.eval('tk::PlaceWindow . center')

        text_length = len(message)
        width = min(400, 20 * text_length)
        height = min(200, 30 * (text_length // 50 + 1)) + 50
        popup.geometry(f"{width}x{height}")

        label = tkinter.Label(popup, text=message, padx=20, pady=20, anchor="w")
        label.pack()

        ok_button = tkinter.Button(popup, text="OK", command=popup.destroy, width=20)
        ok_button.pack(pady=10)

        popup.mainloop()