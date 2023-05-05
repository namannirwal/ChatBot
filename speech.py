import speech_recognition as sr

r= sr.Recognizer()

# It is used to recognize our audio

def voicetotext():
    with sr.Microphone() as source:        # microphone is audio input as source
        print("speak Anything")
        r.adjust_for_ambient_noise(source,duration = 1)
        audio = r.listen(source)
        
        # It will listen to the source and save it in audio.
        try:
            text=r.recognize_google(audio,language='eng-in')
            #convert audio into text
            print("You said: {}".format(text))
            return text
        except:
            text = "Hello"
            return text
   
    
    


