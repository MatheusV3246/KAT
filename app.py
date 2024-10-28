import os
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wavfile
from gtts import gTTS
from dotenv import load_dotenv
import pygame
import threading
import tkinter as tk
from tkinter import ttk
import asyncio
import speech_recognition as sr
from langchain_groq import ChatGroq
from faster_whisper import WhisperModel
from time import sleep
from PIL import Image
import pystray  # Biblioteca para o ícone de bandeja

# Carrega o modelo Whisper uma vez
modelo_whisper = WhisperModel("tiny", compute_type="int8", cpu_threads=os.cpu_count(), num_workers=os.cpu_count())
load_dotenv()  # Carrega variáveis de ambiente

class GravadorDeVoz:
    def __init__(self, taxa_amostragem=16000, pasta_audio='audios'):
        self.taxa_amostragem = taxa_amostragem
        self.pasta_audio = pasta_audio
        
        os.makedirs(self.pasta_audio, exist_ok=True)
        self.limpar_dir_audio(self.pasta_audio)
        
        self.caminho_audio = os.path.join(self.pasta_audio, 'gravacao.wav')
        
        self.dados_audio = []
        
        # Instanciação do modelo LLM
        self.llm = ChatGroq(temperature=0.2, model_name="llama3-8b-8192")
        
        pygame.mixer.init()
        self.esta_gravando = False
        self.esta_falando = False
        self.esta_processando = False
        self.reconhecedor = sr.Recognizer()

    def limpar_dir_audio(self, directory):
        for file_name in os.listdir(directory):
            file_path = os.path.join(directory, file_name)
            if os.path.isfile(file_path) and file_name.endswith('.wav') or file_name.endswith('.mp3'):
                try:
                    os.remove(file_path)
                except:
                    pass
                           
    def iniciar_gravacao(self):
        """Inicia a gravação de áudio."""
        self.dados_audio.clear()
        self.stream = sd.InputStream(samplerate=self.taxa_amostragem, channels=1, callback=self.audio_callback)
        self.stream.start()
        self.esta_gravando = True

    def parar_gravacao(self):
        """Para a gravação de áudio."""
        self.stream.stop()
        self.stream.close()
        
        if self.dados_audio:
            wavfile.write(self.caminho_audio, self.taxa_amostragem, np.concatenate(self.dados_audio, axis=0))
            
        self.esta_gravando = False

    def audio_callback(self, indata, frames, time, status):
        if not status:
            self.dados_audio.append(indata.copy())

    async def transcrever_audio(self):
        """Transcreve o áudio gravado usando o modelo Whisper."""
        try:
            segmentos, _ = modelo_whisper.transcribe(self.caminho_audio, language="pt")
            return "".join(segment.text for segment in segmentos).strip()
        
        except Exception:
            return ""

    async def processar_resposta(self, transcricao):
        """Processa a resposta do modelo a partir da transcrição do áudio."""
        texto_resposta = self.llm.invoke(f"""Seu nome é KAT, e você é uma assistente pessoal
                                        Responda de forma curta, porém completa e objetiva. 
                                        Além disso, seja muito educada e cortês: {transcricao}""").content
        return self.texto_para_fala(texto_resposta)

    def texto_para_fala(self, texto):
        """Converte texto em fala e salva como um arquivo de áudio."""
        arquivo_saida = os.path.join(self.pasta_audio, f"saida_{np.random.randint(100000)}.mp3")
        
        try:
            gTTS(text=texto, lang='pt').save(arquivo_saida)
            return arquivo_saida
        except Exception:
            return ""

    async def tocar_audio(self, caminho_arquivo):
        """Toca o arquivo de áudio especificado."""
        pygame.mixer.music.load(caminho_arquivo)
        self.interface.atualizar_status("🔈 Respondendo...")
        pygame.mixer.music.play()

    def ouvir_comandos(self):
        """Fica ouvindo comandos de voz."""
        with sr.Microphone() as source:
            self.reconhecedor.adjust_for_ambient_noise(source)
            while True:
                try:
                    audio = self.reconhecedor.listen(source)
                    comando = self.reconhecedor.recognize_google(audio, language="pt-BR")

                    if "iniciar" in comando.lower() and not self.esta_gravando and not self.esta_falando and not self.esta_processando:
                        self.iniciar_gravacao()
                        print("Escutando...")
                        self.interface.atualizar_status("👂 Escutando")
                        
                    elif "finalizar" in comando.lower() and self.esta_gravando and not self.esta_falando and not self.esta_processando:
                        self.parar_gravacao()
                        print("Escuta finalizada.")
                        self.interface.atualizar_status("🛑 Escuta finalizada!")
                        sleep(2)
                        asyncio.run(self.processar_transcricao_e_resposta())
                        
                except sr.UnknownValueError:
                    continue
                
                except sr.RequestError as e:
                    print(f"Erro ao se comunicar com o serviço de reconhecimento de fala: {e}")

    async def processar_transcricao_e_resposta(self):
        """Processa a transcrição e a resposta do modelo."""
        self.esta_processando = True
        self.interface.atualizar_layout()
        self.interface.atualizar_status("🔄 Processando...")
        transcricao = await self.transcrever_audio()
        
        if transcricao:
            arquivo_saida = await self.processar_resposta(transcricao)
            if arquivo_saida:
                self.esta_processando = False
                self.esta_falando = True
                self.interface.atualizar_layout()
                
                self.interface.atualizar_status("🔈 Respondendo...")
                await self.tocar_audio(arquivo_saida)
                self.esta_falando = False
                sleep(2)
                self.limpar_dir_audio(self.pasta_audio)
                self.interface.atualizar_layout()
                sleep(1)
                self.interface.atualizar_status("💻 Diga 'Iniciar' para começar!")

class InterfaceGravadorDeVoz:
    def __init__(self, root):
        self.gravador = GravadorDeVoz()
        self.gravador.interface = self
        self.root = root
        self.root.title("KAT | Ket's Assistant Transformer")
        self.root.iconbitmap("images/icon.ico")
        self.configurar_estilos()

        # Configura para ocultar a janela ao fechar
        self.root.protocol("WM_DELETE_WINDOW", self.ocultar_janela)

        # Layout inicial dos componentes
        self.spinner = ttk.Progressbar(root, mode='indeterminate', style="TProgressbar")
        self.botao_cancelar = ttk.Button(root, text="Cancelar Resposta!", command=self.cancelar_reproducao)
        
        # Rótulo de Status
        self.status_label = ttk.Label(root, font=("Arial", 16), foreground="blue")
        self.status_label.pack(pady=30)
        self.atualizar_status("💻 Diga 'Iniciar' para começar!")

        # Atualiza o layout inicial
        self.atualizar_layout()

        # Inicia a escuta em uma nova thread
        threading.Thread(target=self.gravador.ouvir_comandos, daemon=True).start()

        # Adiciona o ícone de bandeja do sistema
        self.iniciar_icone_bandeja()

    def iniciar_icone_bandeja(self):
        """Configura o ícone de bandeja do sistema (System Tray)."""
        # Carrega o ícone a partir do arquivo
        image = Image.open("images/icon.ico")

        # Função para fechar o aplicativo a partir do menu do ícone de bandeja
        def on_quit(icon, item):
            icon.stop()
            self.root.quit()

        # Define o ícone da bandeja com um menu para sair
        self.icone_bandeja = pystray.Icon("KAT Assistant", image, menu=pystray.Menu(
            pystray.MenuItem("Abrir", lambda: self.root.deiconify()),
            pystray.MenuItem("Sair", on_quit)
        ))

        # Executa o ícone de bandeja em uma nova thread
        threading.Thread(target=self.icone_bandeja.run, daemon=True).start()

    def ocultar_janela(self):
        """Oculta a janela em vez de fechá-la."""
        self.root.withdraw()

    def atualizar_layout(self):
        """Atualiza a interface de acordo com o estado atual do Gravador."""
        if self.gravador.esta_processando:
            self.spinner.pack(pady=(10, 20), padx=20)
            self.spinner.start()
        else:
            self.spinner.stop()
            self.spinner.pack_forget()

        if self.gravador.esta_falando:
            self.botao_cancelar.pack(pady=(0, 10))
        else:
            self.botao_cancelar.pack_forget()

        self.root.update_idletasks()

    def configurar_estilos(self):
        """Configura os estilos da interface."""
        estilo = ttk.Style()
        estilo.theme_use("clam")
        estilo.configure("TButton", font=("Helvetica", 12), padding=5)
        estilo.configure("TLabel", font=("Arial", 16), padding=5)
        estilo.configure("TProgressbar", thickness=10)

    def atualizar_status(self, texto):
        """Atualiza o texto do rótulo de status na interface."""
        self.status_label.config(text=texto)
        self.root.update()

    def cancelar_reproducao(self):
        """Cancela a reprodução do áudio."""
        pygame.mixer.music.stop()
        self.atualizar_status("🔈 Resposta cancelada!")
        self.gravador.esta_falando = False
        self.atualizar_layout()
        sleep(2)
        self.atualizar_status("💻 Diga 'Iniciar' para começar!")

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("350x220")
    app = InterfaceGravadorDeVoz(root)
    
    # Minimiza a janela ao iniciar
    root.withdraw()
    
    root.mainloop()
