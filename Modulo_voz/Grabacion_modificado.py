import pyaudio # pip install pyaudio
import wave
import numpy as np
import threading
import keyboard  # pip install keyboard
from google.cloud import speech  # pip install google-cloud-speech
from google.cloud import speech_v1 as speech # pip install google-cloud-translate google-cloud-texttospeech
import os

# configurar credenciales de Google Cloud
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ""

def transcribir_audio_desde_datos(audio_datos, tasa_muestreo=44100, idioma="es-ES"):
    from google.cloud import speech

    client = speech.SpeechClient()

    # Configuración de la solicitud de reconocimiento
    audio = speech.RecognitionAudio(content=audio_datos)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=tasa_muestreo,
        language_code=idioma,
    )

    # Enviar la solicitud para transcribir
    response = client.recognize(config=config, audio=audio)

    # Procesar la respuesta
    transcripciones = []
    for resultado in response.results:
        transcripciones.append(resultado.alternatives[0].transcript)

    texto_transcrito = ' '.join(transcripciones)
    print(f"Transcripción: {texto_transcrito}")
    return texto_transcrito



def grabar_audio_continuo(tasa_muestreo=44100, canales=1, formato=pyaudio.paInt16, duracion_silencio=2):
    # Crear una instancia de PyAudio
    p = pyaudio.PyAudio()

    # Configurar el flujo de audio
    stream = p.open(format=formato,
                    channels=canales,
                    rate=tasa_muestreo,
                    input=True,
                    frames_per_buffer=1024)

    print("Iniciando grabación continua. Presiona 'q' para detener.")
    grabando = True
    frames = []
    silencio_duracion_actual = 0  # Para contar el tiempo de silencio

    # Calcular umbral de ruido usando los primeros dos segundos de grabación
    print("Calculando umbral de ruido...")
    muestras_ruido = []
    for _ in range(int(tasa_muestreo / 1024 * 2)):
        data = stream.read(1024, exception_on_overflow=False)
        audio_np = np.frombuffer(data, dtype=np.int16)
        muestras_ruido.append(np.sqrt(np.mean(audio_np**2)))  # RMS del audio
    umbral_volumen = np.mean(muestras_ruido)
    print(f"Umbral de ruido calculado: {umbral_volumen}")

    def detener_grabacion():
        nonlocal grabando
        while grabando:
            if keyboard.is_pressed('q'):
                print("Deteniendo grabación...")
                grabando = False
                break

    # Iniciar un hilo para monitorear la tecla 'q'
    hilo_tecla = threading.Thread(target=detener_grabacion)
    hilo_tecla.start()

    def iniciar_transcripcion(frames_audio):
        def tarea_transcripcion():
            print("Transcribiendo audio...")
            audio_datos = b''.join(frames_audio)  # Convertir frames a bytes
            texto_transcrito = transcribir_audio_desde_datos(audio_datos)
            print(f"Texto transcrito: {texto_transcrito}")

        hilo_transcripcion = threading.Thread(target=tarea_transcripcion)
        hilo_transcripcion.start()

    while grabando:
        data = stream.read(1024, exception_on_overflow=False)
        frames.append(data)

        # Calcular nivel de volumen
        audio_np = np.frombuffer(data, dtype=np.int16)
        nivel_volumen = np.sqrt(np.mean(audio_np**2))  # RMS del audio

        if nivel_volumen < umbral_volumen:
            silencio_duracion_actual += 1024 / tasa_muestreo
        else:
            silencio_duracion_actual = 0

        # Procesar el audio si se detecta silencio prolongado
        if silencio_duracion_actual >= duracion_silencio:
            audio_total = np.frombuffer(b''.join(frames), dtype=np.int16)
            rms_total = np.sqrt(np.mean(audio_total**2))  # RMS del archivo completo

            if rms_total >= umbral_volumen:
                print("Audio válido detectado, iniciando transcripción...")
                iniciar_transcripcion(frames)
            else:
                print("Segmento descartado por contener solo silencio o ruido de fondo.")

            frames = []  # Reiniciar para el siguiente segmento
            silencio_duracion_actual = 0

    # Procesar cualquier audio restante
    if frames:
        audio_total = np.frombuffer(b''.join(frames), dtype=np.int16)
        rms_total = np.sqrt(np.mean(audio_total**2))

        if rms_total >= umbral_volumen:
            print("Procesando audio final...")
            iniciar_transcripcion(frames)
        else:
            print("Segmento final descartado por contener solo silencio o ruido de fondo.")

    # Detener y cerrar el flujo de audio
    stream.stop_stream()
    stream.close()
    p.terminate()

    print("Grabación finalizada.")


    # Llamada a la función
grabar_audio_continuo()