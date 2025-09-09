# ==============================================================================
# FELIX AI SOLUTIONS - API DEL CANDIDATO DIGITAL (VERSIÓN FINAL)
# Backend con Flask, Control Explícito de Embeddings y Conexión HTTPX
# ==============================================================================

import os
import re
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions # Importamos la utilidad de ChromaDB
from dotenv import load_dotenv
import httpx # Importamos la librería para manejar las conexiones HTTP

# --- 1. CONFIGURACIÓN E INICIALIZACIÓN ---

# Carga las variables de entorno desde el archivo .env (para pruebas locales)
# En Render, estas variables se configuran en el dashboard.
load_dotenv()

# Inicializa la aplicación Flask
app = Flask(__name__)

# Obtiene la clave API de OpenAI de las variables de entorno
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("No se encontró la Clave API de OpenAI. Asegúrate de que tus variables de entorno están configuradas.")

# --- LA SOLUCIÓN DEFINITIVA AL ERROR DE 'PROXIES' ---
# Creamos el cliente de OpenAI pasándole explícitamente un cliente HTTP.
# Esto le ordena a la librería de OpenAI que maneje sus propias conexiones a internet,
# ignorando cualquier configuración de red (proxies) que el entorno de Render
# intente inyectar, que era la causa del error.
client_openai = OpenAI(api_key=OPENAI_API_KEY, http_client=httpx.Client())

# --- CONFIGURACIÓN DE CHROMA CON CONTROL EXPLÍCITO ---
# Le decimos a ChromaDB que debe usar la función de embedding de OpenAI.
# Esto elimina la dependencia de 'onnxruntime' y asegura que todo el sistema
# use el mismo modelo para crear vectores, aumentando la precisión.
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=OPENAI_API_KEY,
                model_name="text-embedding-ada-002"
            )

# Creamos un cliente de base de datos persistente. En Render, esto creará una
# carpeta 'chroma_db' en el disco del servidor para guardar el cerebro de la IA.
client_chroma = chromadb.PersistentClient(path="./chroma_db")

# Obtenemos o creamos la colección (la "tabla" de nuestra base de datos)
# especificando la función de embedding que debe usar.
collection = client_chroma.get_or_create_collection(
                name="candidato_ia_final", # Nuevo nombre para forzar recarga
                embedding_function=openai_ef
            )

# --- 2. LÓGICA DE CARGA DEL CEREBRO ---

def cargar_y_verificar_cerebro():
    """
    Esta función se ejecuta al inicio para cargar el documento de campaña en la base de datos vectorial.
    Compara el número de fragmentos del documento con los vectores en la base de datos
    y la recarga solo si es necesario.
    """
    print("Verificando el estado del cerebro del candidato...")
    campaign_document = """
    ________________________________________

Documento Maestro de Campaña: Javier Montoya Gobernación

Fecha: 25 de agosto de 2025

De: Jefatura de Campaña

Asunto: Perfil Oficial y Plataforma de Gobierno del Candidato Javier Montoya

________________________________________

Perfil del Candidato

Nombre: Javier Montoya

Partido Político: Renovación Quindiana (RQ)

Eslogan de Campaña: Quindío: Nuestro Origen, Nuestro Futuro.

Biografía: Un Hombre de Nuestra Tierra

Javier Montoya no es un político que descubrió el Quindío en un mapa; es un hombre que lleva el aroma de su café y el verde de sus montañas en el alma. Nacido en Calarcá y criado en una finca familiar en las laderas de Córdoba, Javier aprendió desde niño el valor del trabajo arduo y el profundo respeto por la tierra que nos da todo. Estudió Ingeniería Agroindustrial en la Universidad del Quindío, una decisión que tomó para encontrar nuevas formas de agregar valor a los productos de nuestra región y asegurar que las familias campesinas, como la suya, tuvieran un futuro próspero.

Su carrera es un reflejo de su compromiso. En lugar de buscar oportunidades en las grandes capitales, Javier se dedicó a fortalecer el Quindío desde adentro. Fundó una pequeña empresa de cafés especiales que hoy exporta a tres continentes, demostrando que la innovación y la tradición pueden ir de la mano. Posteriormente, sirvió como Secretario de Desarrollo Económico de Armenia, donde lideró la creación de programas de apoyo a emprendedores y atrajo inversión que generó más de 1.500 empleos directos. Javier no es un político tradicional; es un gestor, un quindiano que conoce los problemas del departamento porque los ha vivido y ha trabajado incansablemente para solucionarlos. Vuelve ahora a la arena pública con la experiencia, la visión y la determinación para llevar al Quindío a un nuevo nivel de desarrollo y bienestar.

________________________________________

Pilares del Plan de Gobierno

Nuestra visión para el Quindío se sostiene sobre tres pilares fundamentales, diseñados a partir de las necesidades reales de nuestra gente y el potencial inmenso de nuestra tierra.

Pilar 1: Quindío Próspero y Conectado 

El futuro económico del Quindío depende de nuestra capacidad para innovar sin perder nuestra esencia. No podemos seguir dependiendo de los vaivenes de un solo sector. Mi plan es diversificar y fortalecer nuestra economía, asegurando que la prosperidad llegue a cada rincón del departamento, desde Armenia hasta Génova.

¿Cómo lo haremos?

Centro de Innovación Agroindustrial: Crearemos un centro tecnológico en alianza con la universidad y el sector privado para desarrollar productos con valor agregado a partir del café, el plátano, los cítricos y otros cultivos locales. No más venta de materia prima a bajo costo; transformaremos nuestros productos aquí.

Programa "Caminos del Café": Invertiremos una cifra histórica en la recuperación y pavimentación de las vías terciarias. Un campesino que puede sacar su cosecha de forma rápida y segura es un campesino que prospera. Conectaremos las fincas con los mercados y los corredores turísticos.

Conectividad Digital Rural: Llevaremos internet de alta velocidad a las zonas rurales para potenciar el agroturismo, la educación a distancia y facilitar que nuestros jóvenes puedan trabajar y emprender desde sus municipios.

Pilar 2: Corazón Verde y Seguro 

El Quindío es el corazón verde de Colombia, y nuestro deber es protegerlo como el tesoro que es. Al mismo tiempo, nuestros ciudadanos merecen vivir sin miedo, en calles y barrios seguros. Este pilar integra la defensa de nuestro patrimonio natural con una estrategia frontal contra la delincuencia, porque no hay calidad de vida si falta la tranquilidad o el aire puro.

¿Cómo lo haremos?

Defensa del Territorio: Mi posición es clara e innegociable: No a la megaminería en Salento ni en ningún municipio que ponga en riesgo nuestras fuentes de agua y nuestro Paisaje Cultural Cafetero. Fortaleceremos la autoridad ambiental (CRQ) y crearemos un fondo de "pago por servicios ambientales" para que los campesinos que conservan los bosques sean recompensados.

Seguridad Inteligente y Comunitaria: Implementaremos un sistema de vigilancia con drones para las zonas rurales y perímetros urbanos, disuadiendo el abigeato y el hurto. Crearemos una App ciudadana de "Alerta Temprana" conectada directamente con cuadrantes de la policía y fortaleceremos los Frentes de Seguridad Ciudadana con tecnología y comunicación.

Lucha Frontal contra el Microtráfico: Crearemos una unidad especial de la policía, en coordinación con la fiscalía, dedicada exclusivamente a desmantelar las redes de microtráfico que envenenan a nuestros jóvenes en los barrios de Armenia, Calarcá y Montenegro.

Pilar 3: Oportunidades para Nuestra Gente 

El mayor activo del Quindío es su gente. Sin embargo, por años hemos visto cómo nuestros jóvenes, faltos de oportunidades, deben abandonar la región. Mi obsesión será crear las condiciones para que el talento quindiano se quede, crezca y triunfe aquí. La educación y el empleo de calidad no son un lujo, son la base de nuestro futuro.

¿Cómo lo haremos?

Programa "Mi Primer Empleo Quindiano": Ofreceremos incentivos fiscales significativos a las empresas que contraten a jóvenes recién egresados de universidades e instituciones técnicas. La condición será un contrato a término indefinido con todas las prestaciones de ley.

Bilingüismo y Pertinencia Educativa: En alianza con el SENA y las secretarías de educación, lanzaremos un plan masivo de bilingüismo enfocado en el turismo y la tecnología. Ajustaremos los programas técnicos y tecnológicos para que respondan a la demanda real del sector productivo del Quindío.

Fondo "Emprende Quindío": Destinaremos un capital semilla anual para financiar los 100 mejores proyectos de emprendimiento de base tecnológica, turística y creativa del departamento, con acompañamiento técnico y mentoría para asegurar su éxito.

________________________________________

Preguntas Frecuentes (FAQ)

¿Cuál es su postura clara sobre la minería en Salento y el Valle de Cocora?

Mi postura es de una defensa férrea e innegociable de nuestro patrimonio. Como gobernador, utilizaré todas las herramientas legales y políticas para impedir cualquier proyecto de megaminería que amenace el Valle de Cocora, nuestras fuentes hídricas o la vocación turística y agrícola de Salento. Nuestra riqueza es verde, no dorada. Protegeré el Paisaje Cultural Cafetero como nuestro mayor activo.

El Quindío tiene una de las tasas de desempleo más altas. ¿Cómo va a generar empleo real y rápido?

El empleo no se genera por decreto. Mi estrategia ataca el problema desde tres frentes: 1) Impulso a la construcción a través de la agilización de licencias y el plan de vías terciarias. 2) Incentivos directos a las empresas para que contraten jóvenes a través del programa "Mi Primer Empleo Quindiano". 3) Fortalecimiento del turismo y la agroindustria, que son nuestros mayores generadores de empleo, con mejor infraestructura y acceso a financiación.

¿De dónde saldrá el dinero para financiar todas estas propuestas?

Seremos responsables y eficientes. La financiación provendrá de cuatro fuentes principales: 1) Optimización del gasto público, eliminando burocracia innecesaria y combatiendo la corrupción. 2) Gestión de recursos del Gobierno Nacional, presentando proyectos bien estructurados ante los ministerios. 3) Alianzas Público-Privadas (APP) para grandes obras de infraestructura. 4) Acceso a recursos de cooperación internacional para proyectos de sostenibilidad y desarrollo social.

¿Qué experiencia tiene usted en el sector público?

Mi experiencia es la combinación ideal: conozco las dificultades del sector privado como empresario y entiendo el funcionamiento del Estado desde mi paso por la Secretaría de Desarrollo Económico. No soy un político de escritorio; soy un gestor que ha dado resultados medibles, como la creación de más de 1.500 empleos. Sé cómo ejecutar un presupuesto, cómo liderar equipos y, lo más importante, cómo transformar las ideas en realidades que beneficien a la gente.

El problema del microtráfico está destruyendo a nuestros jóvenes. ¿Qué hará más allá de aumentar el pie de fuerza?

Es un problema con dos caras: la oferta y la demanda. A la oferta la combatiremos con inteligencia y contundencia, con la unidad especial que he propuesto. Pero la clave está en la demanda: invertiremos en programas de prevención del consumo en colegios, fortaleceremos los centros de rehabilitación y crearemos una red de oportunidades (deporte, cultura, empleo) para que nuestros jóvenes vean un futuro más atractivo que el que les ofrecen las drogas.

¿Por qué debería votar por usted y no por los otros candidatos que llevan años en la política?

Porque el Quindío no puede permitirse seguir haciendo lo mismo y esperar resultados diferentes. Los políticos tradicionales ya tuvieron su oportunidad. Yo no ofrezco promesas vacías, ofrezco un plan concreto basado en mi experiencia como emprendedor y como gestor público. No vivo de la política, vivo para servir al Quindío. Mi compromiso es con el futuro de nuestra gente, no con las maquinarias políticas del pasado.

¿Cómo piensa apoyar a los cafeteros y agricultores que se sienten olvidados?

Yo soy uno de ellos, conozco sus luchas. Mi apoyo será total. El programa "Caminos del Café" es una respuesta directa a su necesidad de mejores vías. El Centro de Innovación Agroindustrial les dará herramientas para vender sus productos a mejor precio. Además, crearemos un programa de asistencia técnica para la adaptación al cambio climático y lucharemos por precios justos y acceso a créditos blandos.

El turismo se concentra en Salento y Filandia. ¿Qué hará por los otros municipios?

El turismo debe ser una fuente de ingresos para todo el departamento. Impulsaremos la "Ruta del Café Mágico", una iniciativa que conectará a los municipios de la cordillera como Pijao, Córdoba y Génova, promoviendo el turismo de naturaleza, avistamiento de aves y experiencias cafeteras auténticas. Digitalizaremos la oferta turística de los 12 municipios y la promocionaremos a nivel nacional e internacional.

¿Cuál es su plan para mejorar la salud en el departamento, especialmente en las zonas rurales?

La salud digna es un derecho. Nuestra prioridad será fortalecer la red de hospitales públicos y los puestos de salud en las veredas, garantizando que tengan los médicos, los medicamentos y los equipos básicos. Implementaremos un sistema de telemedicina para que especialistas desde Armenia puedan atender consultas en municipios apartados y crearemos brigadas de salud móviles que lleguen directamente a las fincas.

¿Cómo garantizará la transparencia en su gobierno y luchará contra la corrupción?

Con total determinación. Implementaremos una plataforma de "Gobierno Abierto" donde todos los contratos y presupuestos serán públicos y accesibles en línea para cualquier ciudadano. Crearemos una "Oficina Anticorrupción" dependiente directamente del despacho del gobernador y estableceremos canales de denuncia anónimos y seguros. En mi gobierno, el que le robe un peso al Quindío se va para la cárcel.

¿Cómo trabajará con los alcaldes del departamento? Mi gobierno será un aliado, no un jefe, para los alcaldes. Crearemos mesas de trabajo intermunicipales permanentes para coordinar proyectos regionales en temas de seguridad, infraestructura y desarrollo económico. Promoveremos la unión de esfuerzos para que el progreso no se concentre en la capital, sino que llegue a cada rincón del Quindío.



Seguridad y Convivencia

Pregunta: La percepción de inseguridad en barrios de Armenia y Calarcá es alta. ¿Cómo piensa usted fortalecer la seguridad en zonas urbanas específicas? Respuesta: En Armenia, iniciaremos con un plan piloto en 5 barrios críticos (ej. La Cabaña, La Patria, Miraflores) y en Calarcá, en 3 (ej. El Cacique, Giraldo). Invertiremos $2.500 millones en la instalación de 500 nuevas cámaras de seguridad de alta definición con análisis de video para detección de comportamientos sospechosos. Este sistema estará conectado en tiempo real al Centro de Comando, Control y Comunicaciones (C4), que será reactivado y dotado de personal técnico.

Pregunta: ¿Cuál es su estrategia para combatir la extorsión a pequeños comerciantes? Respuesta: Crearemos una "Mesa Técnica Anti-Extorsión" con la participación quincenal del Gaula, la Policía Judicial y representantes de la Cámara de Comercio. Implementaremos un canal de denuncia confidencial a través de una aplicación móvil con encriptación, para que los comerciantes puedan reportar de forma segura. El objetivo es reducir la tasa de no denuncia en un 40% en los primeros 12 meses.

Pregunta: ¿Cómo abordará la violencia intrafamiliar y la seguridad de las mujeres en el departamento? Respuesta: Destinaremos $1.200 millones a la creación de dos "Casas Refugio" en Armenia y Calarcá, con capacidad para 15 mujeres y sus hijos, respectivamente. Estos centros contarán con psicólogos, abogados y trabajadores sociales. Además, implementaremos el programa "Red de Mujeres Vigilantes", que consistirá en capacitaciones sobre autodefensa y primeros auxilios psicológicos.

________________________________________

Economía y Empleo

Pregunta: El Quindío depende del café y el turismo. ¿Qué hará para diversificar la economía y reducir esa dependencia? Respuesta: Crearemos el "Pacto por la Diversificación Productiva". Invertiremos $3.000 millones en la creación del "Quindío Tech Hub", un espacio físico y digital que ofrecerá incentivos fiscales (exención de impuesto de industria y comercio por 5 años) a 20 empresas de software y tecnología que se instalen en el departamento y contraten al menos a 10 profesionales locales. Adicionalmente, crearemos un programa de formación técnica en codificación y analítica de datos en alianza con el SENA para 500 jóvenes.

Pregunta: ¿Cómo apoyará a los pequeños y medianos empresarios (PyMEs) del Quindío? Respuesta: Lanzaremos el programa "Impulso Quindío", un fondo de capital semilla de $2.000 millones. Este fondo otorgará créditos blandos (tasa de interés del IPC + 2%) con un periodo de gracia de 6 meses para proyectos que demuestren viabilidad y generen al menos 3 empleos formales. La primera convocatoria se abrirá a los 90 días de iniciar el gobierno.

Pregunta: Muchos jóvenes se van del Quindío en busca de oportunidades. ¿Cómo los retendrá? Respuesta: Implementaremos el "Programa de Retención de Talento Joven". Firmaremos acuerdos con 100 empresas del departamento para que ofrezcan 300 pasantías remuneradas anuales a estudiantes universitarios de la región. Quienes decidan emprender, tendrán acceso prioritario al fondo "Mi Primer Emprendimiento".

Pregunta: ¿Qué opina sobre la formalización del empleo y la lucha contra el trabajo informal? Respuesta: Propondremos un subsidio a las PyMEs que formalicen a sus trabajadores. La propuesta es que el gobierno departamental cubra el 50% de los aportes a seguridad social (salud y pensión) durante el primer año de formalización de cada empleado, con un tope de 20 empleados por empresa.

Pregunta: ¿Cuál es su visión para el Eje Cafetero como un bloque económico regional? Respuesta: Convocaremos a los gobernadores de Caldas y Risaralda para firmar un "Acuerdo de Competitividad Regional". Mi propuesta específica es la creación de una oficina conjunta de promoción de inversiones para atraer proyectos de infraestructura turística, tecnológica y agroindustrial a gran escala, y así dejar de competir entre nosotros y comenzar a cooperar.

________________________________________

Educación y Cultura

Pregunta: ¿Cómo mejorará la calidad de la educación pública en el Quindío? Respuesta: Renovaremos 20 aulas en 10 instituciones educativas de zonas rurales y las dotaremos con tabletas y conexión satelital para el acceso a plataformas de educación digital. Firmaremos un convenio con el SENA para implementar el programa "Técnico en Mis Manos", que permitirá a 1.000 estudiantes de los grados 10 y 11 obtener una doble titulación técnica en áreas como agroturismo o energías renovables.

Pregunta: ¿Qué hará para fomentar las artes y la cultura local? Respuesta: Reactivaremos el "Fondo Departamental de Estímulos" con un presupuesto de $800 millones anuales. Este fondo entregará becas a 50 jóvenes artistas para que realicen estudios especializados y financiará 30 proyectos culturales locales, incluyendo festivales y la recuperación de espacios públicos como teatros y casas de la cultura.

Pregunta: ¿Cuál es su postura sobre el deporte en el departamento? ¿Qué hará para apoyar a los deportistas? Respuesta: Crearemos el "Plan de Alto Rendimiento Quindiano". Este programa otorgará 50 becas deportivas de $1.5 millones mensuales a deportistas con potencial para que se dediquen de tiempo completo a su entrenamiento. Adicionalmente, invertiremos $1.800 millones en la adecuación del complejo deportivo de Armenia y en la construcción de 2 canchas sintéticas en municipios con menos de 30.000 habitantes.

________________________________________

Salud

Pregunta: El sistema de salud en el Quindío enfrenta problemas de citas y atención. ¿Cómo lo resolverá? Respuesta: La solución es la digitalización. Implementaremos una plataforma unificada de citas médicas que permitirá a los pacientes agendar, cancelar y reprogramar sus citas desde su teléfono móvil o computadora. El objetivo es reducir el tiempo de espera para citas de medicina general a 48 horas y para especialistas a 15 días.

Pregunta: ¿Qué hará para enfrentar la salud mental en el departamento? Respuesta: Crearemos la "Red de Apoyo Psicológico Departamental". Esta red contará con 50 psicólogos adscritos a los centros de salud que brindarán atención gratuita. Además, en alianza con universidades, ofreceremos talleres de gestión emocional en colegios y empresas.

________________________________________

Infraestructura y Medio Ambiente

Pregunta: ¿Cómo planea mejorar el estado de las vías rurales, que son vitales para el sector agrícola? Respuesta: Destinaremos un fondo de $5.000 millones para el mejoramiento de vías terciarias, con un cronograma trimestral público. Usaremos un modelo de trabajo comunitario en el que la Gobernación aportará la maquinaria y el material, y las comunidades aportarán la mano de obra, agilizando la ejecución. Priorizaremos las vías que conectan a Salento, Filandia y Córdoba, por su impacto turístico y agrícola.

Pregunta: ¿Qué hará para enfrentar el problema de la gestión de residuos sólidos y el saneamiento básico? Respuesta: Financiaremos la construcción de 2 centros de acopio de materiales reciclables en Armenia y Calarcá, en alianza con las asociaciones de recicladores de oficio. El objetivo es aumentar la tasa de reciclaje del 10% al 25% en 4 años.

Pregunta: La protección de los ríos y fuentes de agua es crucial en el Quindío. ¿Qué medidas tomará para proteger los ecosistemas hídricos? Respuesta: Declararemos la cuenca del río Quindío y el río La Vieja como "áreas estratégicas de protección ambiental". Lideraremos un programa de reforestación masiva, sembrando 100.000 árboles nativos en las orillas de estos ríos en los primeros 2 años.

Pregunta: ¿Cuál es su propuesta para la movilidad urbana en Armenia, especialmente ante la congestión del tráfico? Respuesta: Lanzaremos un plan de movilidad multimodal. Construiremos 10 km de ciclorrutas interconectadas en el centro y norte de Armenia, y modernizaremos 30 semáforos con tecnología "inteligente" para optimizar el flujo vehicular en las horas pico.

________________________________________

Corrupción y Gobernanza

Pregunta: La corrupción es un problema recurrente. ¿Qué mecanismo implementará para garantizar la transparencia de su gestión? Respuesta: Implementaremos el "Sistema de Transparencia 360". Cada contrato superior a $50 millones será publicado en línea, con seguimiento en tiempo real del estado de ejecución, las facturas y los informes de interventoría. La ciudadanía podrá hacer comentarios y denuncias directamente en la plataforma, las cuales serán revisadas por un equipo especial de la Oficina de Transparencia.

Pregunta: ¿Cómo garantizará que los cargos públicos se asignen por mérito y no por favores políticos? Respuesta: Crearemos el "Banco de Talento Público del Quindío". Los perfiles de los candidatos a cargos de libre nombramiento y remoción serán evaluados por una comisión externa, que garantizará que los nombramientos se hagan con base en la experiencia, la trayectoria y la idoneidad profesional, y no en la cercanía política.

________________________________________

Relaciones Interinstitucionales y Ciudadanía

Pregunta: ¿Cómo planea trabajar con el Gobierno Nacional para traer recursos al Quindío? Respuesta: Crearemos una "Oficina de Proyectos Estratégicos" que se encargará de identificar y formular proyectos viables que cumplan con los lineamientos del Plan Nacional de Desarrollo. Con esto, buscaremos una inversión de al menos $150.000 millones en los primeros 2 años para proyectos de infraestructura vial y turística.

Pregunta: ¿Qué espacio tendrán los jóvenes y las mujeres en su gobierno? Respuesta: Crearemos el "Gabinete Joven" y el "Gabinete Femenino", con reuniones trimestrales directas conmigo para la formulación de políticas públicas. El 40% de los cargos de dirección y coordinación en mi gobierno estarán ocupados por mujeres y jóvenes menores de 35 años.

Pregunta: ¿Cuál es su mensaje para los quindianos que viven en el exterior y quieren contribuir al desarrollo del departamento? Respuesta: Abriremos la "Ventana Quindiana", una plataforma digital para que los quindianos en el exterior puedan invertir en proyectos de desarrollo local, ya sea a través de capital semilla o compartiendo su conocimiento a través de mentorías virtuales a emprendedores.

Pregunta: ¿Cómo fomentará la participación ciudadana en su administración? Respuesta: Realizaremos un "Diálogo de Gobierno" anual en cada uno de los 12 municipios, donde presentaré los avances de mi gestión y responderé directamente a las preguntas de la ciudadanía. Todos los planes de inversión serán sometidos a audiencias públicas.

Pregunta: ¿Cuál es su visión sobre el futuro de los municipios de la cordillera, como Salento y Córdoba? Respuesta: Mi visión es de "turismo de baja huella". Apoyaré a los emprendimientos locales de ecoturismo y gastronomía, y trabajaré con los Parques Nacionales para proteger el Valle de Cocora y las reservas naturales, limitando la capacidad de carga para evitar la masificación.

Pregunta: ¿Qué hará para fortalecer la conexión entre la capital (Armenia) y el resto de los municipios? Respuesta: Implementaremos un sistema de transporte público intermunicipal más eficiente. Coordinaremos con las alcaldías para la construcción de una central de transferencia de pasajeros en Armenia. Además, trabajaremos en la mejora de las vías principales que conectan a la capital con los municipios.

________________________________________

Temas Complementarios

Pregunta: ¿Cuál es su postura sobre el futuro de la industria turística post-pandemia? Respuesta: La pandemia nos enseñó a valorar el turismo de naturaleza. Invertiremos $1.000 millones en la promoción del Quindío como un "destino de bienestar", con énfasis en el agroturismo y las experiencias en fincas cafeteras.

Pregunta: ¿Qué opina sobre la relación entre el gobierno y el sector privado? Respuesta: El sector privado es un socio, no un rival. Crearemos una "Mesa de Fomento a la Inversión" que se reunirá mensualmente para identificar cuellos de botella y crear un marco normativo que facilite la creación de empresas y la generación de empleo en el departamento.

Pregunta: ¿Qué propone para el sector rural, más allá de la producción? Respuesta: Implementaremos el programa "Campo Conectado". Trabajaremos con las empresas de telecomunicaciones para llevar conectividad satelital a 100 veredas, facilitando el acceso a educación a distancia, teletrabajo y comercio electrónico para los productores.

Pregunta: ¿Cuál es su posición frente a la minería ilegal y los cultivos ilícitos? Respuesta: Tolerancia cero. Crearemos un grupo élite con la Policía y el Ejército para desmantelar las estructuras de minería ilegal, con el uso de drones de vigilancia en las zonas críticas. Implementaremos un programa de sustitución voluntaria de cultivos ilícitos, ofreciendo a las familias campesinas alternativas productivas y apoyo técnico.

Pregunta: ¿Qué hará para proyectar la marca "Quindío" a nivel nacional e internacional? Respuesta: Invertiremos $500 millones en una campaña de marketing digital en plataformas como Google y redes sociales para atraer a turistas de Estados Unidos, Canadá y España. La campaña estará centrada en el Paisaje Cultural Cafetero y en las experiencias únicas de la región, destacando al Quindío como el corazón de Colombia.

Preguntas y Respuestas Adicionales

Pregunta: ¿Cuál es su postura frente a la sostenibilidad de las fincas cafeteras familiares ante la fluctuación del precio del café? Respuesta: Defenderemos el precio interno del café. Crearemos un Fondo de Estabilización de Precios de $1.000 millones, capitalizado con aportes departamentales y cooperación internacional, para subsidiar a los pequeños productores cuando el precio por carga de café caiga por debajo del costo de producción.

Pregunta: ¿Cómo piensa impulsar la economía circular y el reciclaje a nivel departamental? Respuesta: Implementaremos un Programa de Incentivos al Reciclaje. Las familias y empresas que certifiquen el 80% de su separación en la fuente recibirán un descuento del 5% en su impuesto predial, y las empresas recicladoras formales obtendrán un subsidio para la compra de maquinaria.

Pregunta: ¿Cuál es su plan para mejorar la calidad de vida de los adultos mayores en el Quindío? Respuesta: Lanzaremos el programa "Mayores Productivos". En alianza con el SENA, ofreceremos talleres de oficios y manualidades. Crearemos un fondo para la comercialización de sus productos en ferias locales y en plataformas digitales, garantizándoles una fuente de ingresos digna.

Pregunta: ¿Cómo enfrentará el desafío de la infraestructura para personas con discapacidad? Respuesta: Exigiremos que en toda obra pública nueva (parques, andenes, edificios) se incorporen rampas, señalización braille y senderos podotáctiles. Destinaremos $500 millones para adecuar los espacios públicos existentes en las cabeceras municipales.

Pregunta: ¿Qué hará para fortalecer la identidad cultural del Paisaje Cultural Cafetero ante la globalización? Respuesta: Impulsaremos la enseñanza del patrimonio cultural cafetero en los colegios a través de una cátedra obligatoria. Financiaré la creación de "Rutas de Sabores y Saberes" para que los turistas puedan interactuar con artesanos, baristas y arrieros.

Pregunta: ¿Cuál es su estrategia para atraer inversiones extranjeras al departamento? Respuesta: Crearemos una oficina de "Ventanilla Única" para la inversión extranjera, que agilizará los trámites de registro de empresas en un 50%. Acompañaremos a los inversionistas en todo el proceso y les ofreceremos beneficios fiscales en zonas de desarrollo prioritario.

Pregunta: ¿Cómo se relacionará con las minorías étnicas y comunidades indígenas presentes en el Quindío? Respuesta: Conformaré la Mesa de Diálogo Étnico, con participación permanente de los líderes indígenas y afrocolombianos. Garantizaremos el respeto a sus territorios ancestrales y co-crearemos políticas públicas para la preservación de su cultura y la mejora de su calidad de vida.

Pregunta: ¿Qué propone para el manejo de los animales domésticos y la protección de la fauna silvestre? Respuesta: Destinaremos $400 millones para construir y operar un Centro de Bienestar Animal en Armenia, que ofrecerá servicios de esterilización gratuita, vacunación y atención veterinaria de bajo costo. Además, fortaleceremos la autoridad ambiental para combatir el tráfico de fauna silvestre.

Pregunta: ¿Qué hará para garantizar la seguridad alimentaria de las familias vulnerables? Respuesta: Fortaleceremos los bancos de alimentos y crearemos la Red de Huertas Comunitarias. Suministraremos capital semilla, herramientas y asistencia técnica a 500 familias vulnerables para que puedan sembrar sus propios alimentos orgánicos.

Pregunta: ¿Cómo abordará el problema de los habitantes de calle en las principales ciudades del departamento? Respuesta: Mi enfoque será integral. En coordinación con la Secretaría de Salud y Bienestar Social, activaremos brigadas móviles que ofrezcan atención psicológica, de salud y programas de resocialización y capacitación para el empleo. No los criminalizaremos, los apoyaremos.

Pregunta: ¿Cuál es su visión para el Aeropuerto Internacional El Edén? Respuesta: Buscaremos convertirlo en un hub logístico y de carga aérea. Impulsaremos la inversión para la expansión de su infraestructura, atrayendo a aerolíneas de bajo costo y de carga, lo que potenciará la exportación de nuestros productos agrícolas.

Pregunta: ¿Cómo impulsará la industria de eventos y congresos en el departamento? Respuesta: Crearemos el "Fondo de Promoción de Eventos" de $500 millones, que cofinanciará la realización de congresos y eventos nacionales e internacionales en el Quindío. Esto dinamizará la hotelería, el comercio y el turismo.

Pregunta: ¿Cuál es su postura frente a la exploración de recursos no renovables en el Quindío? Respuesta: Mi postura es de defensa total de nuestro patrimonio natural y hídrico. No permitiré la exploración ni la explotación de petróleo o minerales en el departamento. Nuestro oro es el agua, el café y el turismo sostenible.

Pregunta: ¿Qué hará para apoyar a las comunidades campesinas que no se dedican al café? Respuesta: Diversificaremos el apoyo agrícola a otros productos como el aguacate, la mora y los cítricos. Crearemos centros de acopio y facilitaremos la comercialización directa con grandes cadenas de supermercados y exportadores, eliminando intermediarios.

Pregunta: ¿Cómo fortalecerá la infraestructura tecnológica y la conectividad en las zonas rurales? Respuesta: En alianza con el Ministerio de las TIC, buscaremos la instalación de 100 puntos de Wi-Fi gratuito en parques y zonas comunes de los municipios y veredas más apartadas.

Pregunta: ¿Qué propone para el desarrollo de la economía naranja en el Quindío? Respuesta: Crearemos la "Incubadora de Emprendimientos Creativos" en alianza con las universidades. Ofreceremos capital semilla, asesoría en propiedad intelectual y mercadeo a los proyectos de cine, música, diseño y artesanías.

Pregunta: ¿Cómo promoverá el Quindío como un destino de turismo de aventura y deportivo? Respuesta: Invertiremos en la adecuación de senderos para senderismo y ciclismo de montaña, con señalización y puntos de asistencia. Organizaremos anualmente un Festival de Deportes de Aventura que atraiga a turistas nacionales e internacionales.

Pregunta: ¿Qué hará para que las instituciones educativas estén más conectadas con el mercado laboral? Respuesta: Conformaremos mesas de trabajo entre rectores universitarios, directores del SENA y gremios empresariales para que la oferta académica responda a las necesidades de la región. El objetivo es que el 80% de los egresados encuentren empleo en los primeros 6 meses.

Pregunta: ¿Cuál es su plan para dignificar el trabajo de los recicladores y de la población que vive del rebusque? Respuesta: Formalizaremos a las cooperativas de recicladores y les brindaremos apoyo técnico y económico para la compra de vehículos y maquinaria. Apoyaremos a los vendedores informales en la creación de asociaciones y les asignaremos espacios de trabajo dignos.

Pregunta: ¿Qué propone para garantizar el acceso a la justicia y los servicios de las entidades públicas en las zonas más apartadas? Respuesta: Crearemos las "Jornadas de Gobierno Móvil". Brigadas de funcionarios de la Gobernación, la Registraduría y la Defensoría del Pueblo se trasladarán a los municipios y veredas más lejanas, llevando servicios como la expedición de documentos, asesoría jurídica y atención social.



5.1. Hoja de Vida y Transparencia Financiera

Javier Montoya es un líder con una trayectoria pública y privada impecable. Su experiencia no es un misterio; es el pilar de su capacidad para gobernar. A continuación, un resumen de su perfil y su compromiso con la transparencia.

Hoja de Vida:

Educación: Ingeniero Agroindustrial, Universidad del Quindío (2000). Especialización en Gerencia de Proyectos, Universidad EAFIT (2006).

Experiencia Laboral:

2007 - 2015: Fundador y Gerente General de "Cafés Especiales Quindío", empresa exportadora que llevó la marca regional a mercados en Europa y Asia.

2016 - 2019: Secretario de Desarrollo Económico de Armenia. Lideró la creación del "Fondo Emprende" y la atracción de 5 empresas de tecnología, generando más de 1.500 empleos.

2020 - 2024: Consultor privado en agronegocios y sostenibilidad.

Declaración de Renta: Javier Montoya ha publicado un resumen de su declaración de renta y bienes de 2024, demostrando su compromiso con la transparencia. Sus ingresos provienen exclusivamente de su actividad profesional como consultor. No posee contratos con el Estado, ni ha sido investigado por ningún órgano de control.

________________________________________

5.2. Aclaraciones y Rumores Comunes

En una campaña limpia, los rumores deben enfrentarse con la verdad. Javier Montoya no tiene nada que ocultar.

Rumor 1: "Javier Montoya es un empresario sin experiencia política".

Aclaración: La experiencia de Javier es la de un gestor. Como Secretario de Desarrollo Económico de Armenia, lideró políticas públicas, gestionó presupuestos de más de $50.000 millones y supervisó a más de 100 funcionarios. Su enfoque no es el de un político de carrera, sino el de un líder que logra resultados.

Rumor 2: "Los recursos de la campaña de Montoya provienen de grandes financiadores externos".

Aclaración: La campaña de Renovación Quindiana está financiada por más de 300 donantes, en su mayoría pequeños y medianos empresarios del Quindío, caficultores, profesionales y ciudadanos que creen en su visión. El 80% de los aportes no superan los 5 millones de pesos, demostrando que este es un proyecto de la gente.

________________________________________

5.3. Testimonios y Apoyos Públicos

La mejor prueba de un buen liderazgo es el respaldo de la gente que lo conoce y ha trabajado con él.

Marcela Pérez, Caficultora de Génova: "Cuando el precio del café se cayó, Javier Montoya fue el único que nos ayudó a crear una marca para venderlo a un mejor precio. Él sabe que nuestro futuro está en el campo, y no en la política de oficina."

Carlos Ochoa, Emprendedor Tecnológico de Armenia: "Gracias a los programas que Javier impulsó cuando fue Secretario, mi empresa de software pudo crecer y hoy empleo a más de 20 jóvenes de la región. Él entiende que la tecnología es el futuro del Quindío."

Laura Torres, Líder Comunitaria de La Tebaida: "Javier no es solo un candidato que viene a prometer. Él ha caminado nuestros barrios, conoce nuestras necesidades y, lo más importante, tiene un plan claro y detallado para la seguridad y el empleo de nuestra gente."

________________________________________

5.4. Sostenibilidad y Finanzas de la Campaña

La transparencia es un pilar de nuestra gestión. Por eso, hemos dispuesto la información sobre el financiamiento y presupuesto de nuestra campaña.

Presupuesto: El presupuesto de la campaña se estima en $1.500 millones, distribuidos de la siguiente manera:

Publicidad y Medios Digitales: 40%

Movilización y Eventos en Territorio: 30%

Logística y Funcionamiento: 20%

Reservas para Imprevistos: 10%

Fuentes de Financiación: La campaña de Renovación Quindiana se financia de manera transparente y ética. Las fuentes de ingreso provienen principalmente de donaciones de personas naturales (80%) y de empresas locales (20%) comprometidas con el desarrollo regional. No aceptamos donaciones de empresas con contratos públicos ni de personas investigadas por la justicia.

________________________________________

5.5. Llamado a la Participación Ciudadana

El Quindío de hoy no se construirá solo con un líder, sino con la participación de todos.

Canales de Contacto:

WhatsApp: +57 320 XXX XXXX (Envía tu pregunta o sugerencia).

Redes Sociales: @JavierMontoyaGobernador (Instagram y Facebook).

Correo Electrónico: contacto@javiermontoya.com

Calendario de Eventos:

2 de septiembre: Encuentro ciudadano en el Parque Bolívar, Armenia.

9 de septiembre: Recorrido por las fincas cafeteras de Génova.

15 de septiembre: Debate abierto con jóvenes en la Universidad del Quindío.

20 de septiembre: Encuentro con el sector turístico en Salento.

Tu apoyo, tu voz y tus ideas son cruciales. Con tu participación, construiremos el Quindío que merecemos.





    """
    
    # Estrategia de Hyper-Chunking: dividimos el documento en oraciones para máxima precisión.
    text_chunks = re.split(r'(?<=[.?!])\s+', campaign_document.replace('\n', ' ').replace('•', '').replace('¿', '').replace('¡', ''))
    text_chunks = [chunk.strip() for chunk in text_chunks if len(chunk.strip()) > 15]

    # Comparamos si la base de datos está sincronizada con el documento actual.
    if collection.count() != len(text_chunks):
        print(f"La base de datos (con {collection.count()} vectores) no coincide con el documento (con {len(text_chunks)} chunks). Recargando...")
        
        # Si hay datos antiguos, los borramos para una carga limpia.
        if collection.count() > 0:
            ids_to_delete = [f"doc_chunk_{i}" for i in range(collection.count())]
            if ids_to_delete: collection.delete(ids=ids_to_delete)
        
        # Añadimos los nuevos documentos. ChromaDB, con la configuración que le dimos,
        # se encargará automáticamente de convertirlos a embeddings usando la API de OpenAI.
        doc_ids = [f"doc_chunk_{i}" for i in range(len(text_chunks))]
        collection.add(
            documents=text_chunks,
            ids=doc_ids
        )
        print("¡Cerebro recargado con éxito!")
    else:
        print("El cerebro está cargado y actualizado.")

# --- 3. LÓGICA DE IA (NÚCLEO CONVERSACIONAL) ---

def ask_candidato_ia(pregunta):
    """
    Orquesta todo el proceso de respuesta: busca contexto en la DB y genera una respuesta con el LLM.
    """
    contexto = ""
    try:
        # Con un embedding consistente, la búsqueda directa es muy potente.
        # ChromaDB recibe el texto y usa la función de OpenAI internamente para la búsqueda.
        results = collection.query(
            query_texts=[pregunta],
            n_results=10 # Obtenemos un contexto amplio para dar respuestas completas
        )
        contexto = "\n\n".join(results['documents'][0])
    except Exception as e:
        print(f"Error al buscar en ChromaDB: {e}")
        return "Hubo un problema al consultar mi base de conocimiento. Por favor, intenta de nuevo."

    # Usamos el prompt V11, que encontramos que tenía el mejor equilibrio.
    prompt_template = f"""
    Tu Persona: Eres Javier Montoya, candidato a la gobernación del Quindío. Responde siempre en primera persona ("Mi propuesta es...", "Yo creo..."). Tu tono es el de un líder cercano, experto y directo.
    Tu Misión: Dar la respuesta más completa y útil posible a la pregunta del ciudadano, basándote en la "Información de Apoyo" que se te proporciona.
    Instrucciones Clave:
    1.  **Sintetiza y Conecta:** Lee TODA la "Información de Apoyo" y conecta las diferentes ideas para dar una respuesta completa y natural. No te limites a un solo fragmento.
    2.  **AUMENTA CON HECHOS:** Siempre que sea posible, enriquece tu respuesta mencionando nombres de programas específicos, cifras o datos concretos que encuentres en la "Información de Apoyo". Esto hace tu respuesta más fuerte y creíble.
    3.  **GUARDARRAÍL DE REALIDAD:** Construye tu respuesta usando únicamente las ideas y propuestas presentes en la "Información de Apoyo". Puedes elaborar sobre ellas para ser más conversacional, pero no introduzcas programas, planes o promesas nuevas que no estén en el texto.
    4.  **Ignora Errores Menores:** La pregunta del ciudadano puede tener errores de ortografía o gramática. Ignóralos y céntrate en la intención principal de la pregunta.
    5.  **Regla de Oro:** Si la "Información de Apoyo" contiene datos relevantes, ÚSALOS para responder. Si está vacía o no tiene relación alguna con la pregunta, responde: "No estoy seguro de haber entendido bien tu pregunta. ¿Podrías reformularla de una manera más clara para poder darte una respuesta precisa?".
    ---
    Información de Apoyo (Contexto Extraído de tu Base de Datos):
    {contexto}
    ---
    Pregunta del Ciudadano (ignora errores de tipeo):
    "{pregunta}"
    Respuesta de Javier Montoya:
    """
    
    try:
        res_completion = client_openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt_template}],
            temperature=0.4 # La temperatura del equilibrio
        )
        return res_completion.choices[0].message.content
    except Exception as e:
        print(f"Error al generar la respuesta: {e}")
        return "Tuve un inconveniente al formular la respuesta. Por favor, intenta de nuevo."

# --- 4. EL ENDPOINT (LA PUERTA DE ENTRADA PARA TWILIO) ---

@app.route("/whatsapp", methods=['POST'])
def whatsapp_reply():
    """Esta función es llamada por Twilio cada vez que llega un mensaje de WhatsApp."""
    incoming_msg = request.values.get('Body', '').strip()
    print(f"Mensaje recibido: {incoming_msg}")

    # Prepara la respuesta de Twilio
    resp = MessagingResponse()
    
    # Obtiene la respuesta de nuestra IA
    respuesta_ia = ask_candidato_ia(incoming_msg)
    
    # Empaqueta la respuesta para Twilio
    resp.message(respuesta_ia)
    print(f"Respuesta enviada: {respuesta_ia}")

    # Devuelve la respuesta a Twilio para que la envíe por WhatsApp
    return str(resp)

# --- 5. INICIO DE LA APLICACIÓN ---

# Este bloque se ejecuta solo cuando corremos el script directamente (como en Render).
if __name__ == "__main__":
    # Nos aseguramos de que la base de datos esté lista antes de que el servidor empiece a aceptar peticiones.
    cargar_y_verificar_cerebro()
    
    # Obtenemos el puerto que Render nos asigna. Si no lo encuentra, usa 5000 por defecto.
    port = int(os.environ.get("PORT", 5000))
    
    print(f"API iniciada. Escuchando en el puerto {port}")
    
    # Usamos Waitress como servidor de producción. Es simple y compatible con Windows y Linux.
    # Necesitaremos añadir 'waitress' a nuestro requirements.txt
    from waitress import s