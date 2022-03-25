#Teks to speek menggunakan Espeak

import subprocess

def execute_unix(inputcommand):
   p = subprocess.Popen(inputcommand, stdout=subprocess.PIPE, shell=True)
   (output, err) = p.communicate()
   return output

a = "Anda Tidak pakai masker, tidak boleh masuk ruangan. Tolong pakai masker dulu"

# create wav file / TTS menjadi file
# w = 'espeak -w temp.wav "%s" 2>>/dev/null' % a  
# execute_unix(w)

# tts using espeak
c = 'espeak -vid+f2 -k19 -s154 --punct="<characters>" "%s" 2>>/dev/null --stdout|aplay' % a 
execute_unix(c)
