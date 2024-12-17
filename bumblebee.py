# from huggingface_hub import InferenceClient
# from langchain_experimental.agents import create_pandas_dataframe_agent
# from langchain_huggingface import HuggingFaceEndpoint
# import pandas as pd

# # Configurar HuggingFace Client para LangChain
# llm = HuggingFaceEndpoint(
#     endpoint_url="https://api-inference.huggingface.co/models/meta-llama/Llama-3.2-3B-Instruct",
#     temperature=0.7,
#     huggingfacehub_api_token="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"
# )

# # Cargar los datasets
# try:
#     df1 = pd.read_csv('dft_road_casualty_statistics_casualty_UK.csv')
#     df2 = pd.read_csv('dft_road_casualty_statistics_collisions_UK.csv')
#     df3 = pd.read_csv('dft_road_casualty_statistics_vehicle_UK.csv')
#     df4 = pd.read_csv('traffic_density_UK.csv')
#     dataframes = {"dataset1": df1, "dataset2": df2, "dataset3": df3, "dataset4": df4}
# except FileNotFoundError:
#     print("Error: Uno o más archivos CSV no se pudieron cargar.")
#     dataframes = {}

# # Crear agentes para cada DataFrame
# agents = {
#     name: create_pandas_dataframe_agent(llm, df, verbose=True, allow_dangerous_code=True)
#     for name, df in dataframes.items()
# }

# # Función para manejar preguntas sobre los datasets
# def handle_query(query):
#     for name, agent in agents.items():
#         try:
#             print(f"Intentando con {name}...")
#             response = agent.run(query)
#             return f"Respuesta de {name}: {response}"
#         except Exception as e:
#             print(f"Error al procesar con {name}: {e}")
#     return "No se encontró respuesta en ninguno de los datasets. Verifica tu consulta."

# # Chatbot principal
# print("¡Hola! Puedes hacerme preguntas sobre los datasets cargados (escribe 'salir' para terminar).")

# while True:
#     user_input = input("Tú: ")

#     if user_input.lower() == "salir":
#         print("Bot: ¡Adiós! Espero haberte ayudado.")
#         break

#     # Procesar consulta relacionada con los datasets
#     response = handle_query(user_input)
#     print(f"Bot: {response}")





# from huggingface_hub import InferenceClient
# from langchain_experimental.agents import create_pandas_dataframe_agent
# from langchain_huggingface import HuggingFaceEndpoint
# import pandas as pd

# # Configurar HuggingFace Client para LangChain
# llm = HuggingFaceEndpoint(
#     endpoint_url="https://api-inference.huggingface.co/models/meta-llama/Llama-3.2-3B-Instruct",
#     temperature=0.7,
#     huggingfacehub_api_token="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"
# )

# # Cargar los datasets
# try:
#     df1 = pd.read_csv('dft_road_casualty_statistics_casualty_UK.csv')
#     df2 = pd.read_csv('dft_road_casualty_statistics_collisions_UK.csv')
#     df3 = pd.read_csv('dft_road_casualty_statistics_vehicle_UK.csv')
#     df4 = pd.read_csv('traffic_density_UK.csv')
#     dataframes = {
#         "dataset1": df1,
#         "dataset2": df2,
#         "dataset3": df3,
#         "dataset4": df4
#     }
# except FileNotFoundError:
#     print("Error: Uno o más archivos CSV no se pudieron cargar.")
#     dataframes = {}

# # Crear agentes para cada DataFrame
# agents = {
#     name: create_pandas_dataframe_agent(
#         llm,
#         df,
#         verbose=True,
#         allow_dangerous_code=True,  # Habilita ejecución de código peligroso
#         tools=["python_repl_ast"]  # Habilita ejecución de código Python.
#     )
#     for name, df in dataframes.items()
# }

# # Función para manejar preguntas sobre los datasets
# def handle_query(query):
#     for name, agent in agents.items():
#         try:
#             print(f"Intentando con {name}...")
#             response = agent.invoke({"input": query})  # Usar invoke en lugar de run.
#             return f"Respuesta de {name}: {response}"
#         except Exception as e:
#             print(f"Error al procesar con {name}: {e}")
#     return "No se encontró respuesta en ninguno de los datasets. Verifica tu consulta."

# # Chatbot principal
# print("¡Hola! Puedes hacerme preguntas sobre los datasets cargados (escribe 'salir' para terminar).")

# while True:
#     user_input = input("Tú: ")

#     if user_input.lower() == "salir":
#         print("Bot: ¡Adiós! Espero haberte ayudado.")
#         break

#     # Procesar consulta relacionada con los datasets
#     response = handle_query(user_input)
#     print(f"Bot: {response}")








from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_huggingface import HuggingFaceEndpoint
import pandas as pd

# Configuración del modelo HuggingFace
llm = HuggingFaceEndpoint(
    endpoint_url="https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct",
    temperature=0.7,
    huggingfacehub_api_token="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"
)

# Cargar el dataset
try:
    df = pd.read_csv('traffic_density_UK.csv')
    print("Dataset cargado correctamente.")
except FileNotFoundError:
    print("Error: El archivo 'traffic_density_UK.csv' no se encuentra.")
    df = None

# Verificar que el DataFrame no esté vacío
if df is not None:
    # Configurar el agente con límites y herramientas válidas
    try:
        agent = create_pandas_dataframe_agent(
            llm,
            df,
            verbose=True,
            max_iterations=3,  # Límite de iteraciones
            handle_parsing_errors=True,  # Manejo de errores de parsing
            allow_dangerous_code=True    # Permitir ejecución de código Python arbitrario
        )
        print("Agente creado correctamente.")
    except Exception as e:
        print(f"Error al crear el agente: {e}")
        agent = None
else:
    print("No se pudo crear el agente porque el dataset no está disponible.")
    agent = None

# Función para realizar consultas
def handle_query(query):
    if agent is None:
        return "El agente no está disponible debido a problemas con el dataset o la configuración."

    try:
        # Realizar la consulta al agente
        print(f"Consulta: {query}")
        response = agent.run(query)
        return f"Respuesta: {response}"
    except Exception as e:
        return f"Error al procesar la consulta: {e}"

# Interfaz de chat simple
if agent is not None:
    print("¡Hola! Puedes hacerme preguntas sobre el dataset cargado (escribe 'salir' para terminar).")

    while True:
        user_input = input("Tú: ")

        if user_input.lower() == "salir":
            print("Bot: ¡Adiós! Espero haberte ayudado.")
            break

        # Procesar consulta relacionada con el dataset
        response = handle_query(user_input)
        print(f"Bot: {response}")
else:
    print("El sistema no está operativo debido a problemas con el agente o el dataset.")

