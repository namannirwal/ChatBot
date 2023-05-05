from tkinter import *
from chat import get_response,bot_name
from speech import voicetotext

BG_COLOR ="#FFCE30"
TEXT_COLOR ="#000000"

FONT="Helvetica 16"
FONT_BOLD="Helvetica 16 bold"

class ChatApplication:
    def __init__(self):
        self.window= Tk()
        self._setup_main_window()

    def run (self):
        self.window.mainloop()
    
    def _setup_main_window(self):
        self.window.title("BOT")
        self.window.resizable(width=False,height=False)
        self.window.configure(width=800,height=700,bg=BG_COLOR)

        #HEAD LABEL
        head_label=Label(self.window,bg=BG_COLOR,fg=TEXT_COLOR,text="AICTE CHAT-BOT",font=FONT_BOLD,pady=10)
        head_label.place(relwidth=1)  #relwidth set to 1 width will be equal to maxwidth

        #DIVIDER
        line=Label(self.window,width=450,bg="#C25A3D")
        line.place(relwidth=1,rely=0.07,relheight=0.012)

        #TEXT_Widget
        self.text_widget=Text(self.window,width=20,height=2,bg=BG_COLOR,fg=TEXT_COLOR,font=FONT,padx=5,pady=5)
        self.text_widget.place(relwidth=1,rely=0.08,relheight=0.745)
        self.text_widget.configure(cursor="arrow",state=DISABLED) 
       
        #SCROLLBAR
        scrollbar=Scrollbar(self.text_widget)
        scrollbar.place(relheight=1,relx=0.975)
        scrollbar.configure(command=self.text_widget.yview) # change the view of the y axis of the text_widget

        #BOTTOM LABEL
        bottom_label=Label(self.window,bg="#C25A3D",height=80)
        bottom_label.place(relwidth=1,rely=0.825)

        #msg Entry
        self.msg_entry=Entry(bottom_label,bg="#2C3E50",fg=TEXT_COLOR,font=FONT)
        self.msg_entry.place(relwidth=0.74,relheight=0.06,relx=0.011,rely=0.008)
        self.msg_entry.focus()   # automatically selects the message sending box by default
        self.msg_entry.bind("<Return>",self._on_enter_pressed)

        #SEND BUTTON
        btn=Button(bottom_label,text="Send",font=FONT_BOLD,width=20,bg="#C25A3D",command=lambda: self._on_enter_pressed(None))
        btn.place(relx=0.77,rely=0.008,relheight=0.06,relwidth=0.10)

        btn=Button(bottom_label,text="Voice",font=FONT_BOLD,width=20,bg="#C25A3D",command=lambda:self.msgfieldvoice())
        btn.place(relx=0.89,rely=0.008,relheight=0.06,relwidth=0.10)       
    
    def msgfieldvoice(self):
        sentence = voicetotext()
        self.msg_entry.insert(END,sentence)

    def _on_enter_pressed(self,event):
        msg=self.msg_entry.get()
        self._insert_message(msg,"You")


    def _insert_message(self,msg,sender):
        if not msg:
            return
        
        self.msg_entry.delete(0,END)
        msg1=f"{sender}: {msg}\n\n"
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END,msg1)
        self.text_widget.configure(state=DISABLED)

        msg2=f"{bot_name}: {get_response(msg)}\n\n\n"
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END,msg2)
        self.text_widget.configure(state=DISABLED)
       
        self.text_widget.see(END)     # scrolls to the last message
       
if __name__ == "__main__":
    app= ChatApplication()
    app.run()
