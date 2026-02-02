import pandas
import sklearn

"""
	Vamos a entrenar un modelo usando un csv de datos de
	estudiantes y sus habitos de estudio para determinar
	su nota en el examen final.

	Todas las columnas disponibles:

	student_id,
	age,
	country,
	prior_programming_experience,
	weeks_in_course,
	hours_spent_learning_per_week,
	practice_problems_solved,
	projects_completed,
	tutorial_videos_watched,
	uses_kaggle,
	participates_in_discussion_forums,
	debugging_sessions_per_week,
	self_reported_confidence_python,
	final_exam_score,
	passed_exam
"""

data = pandas.read_csv("./data.csv")

"""
	Vamos a limpiar nuestros datos.

	Primero nos quitamos las columnas que (en principio) 
	no nos aportan información útil para clasificar:
		- student_id
		- country
		- passed_exam
	
	Ademas filtramos las filas NaN y otros tipos de nulos en
	la columna "prior_programming_experience".
"""

# Eliminar las columnas "sobrantes" o que no aportan información
data = data.drop(columns=["student_id", "country", "passed_exam"])

# Filtrar las filas donde es NaN u otros tipos de nulos
data = data[data["prior_programming_experience"].notna()]


"""
	Normalizar los datos

	Agrupamos la nota final por etiquetas categoricas en función
	del rango de nota.

	Label encoder para las columnas categoricas que SÍ vamos a usar.
	Al importar el orden Begginer < Intermediate < Advanced se hace
	con label encoder. Lo hago a mano para ordenar como quiero.

	Para el resto de columnas numéricas vamos a "escalarlas"
"""

# Transformar 'final_exam_score' en etiquetas
bins = [0, 49, 69, 89, 100]  # los límites de cada rango
labels = ['suspenso', 'aprobado', 'notable', 'sobresaliente']  # etiquetas

data['final_exam_score'] = pandas.cut(data['final_exam_score'], bins=bins, labels=labels, include_lowest=True)

# Label encoder "a mano" de experiencia
mapping = {
    "Beginner": 0,
    "Intermediate": 1,
    "Advanced": 2
}
data["prior_programming_experience"] = data["prior_programming_experience"].map(mapping)

# Label encoder "a mano" de resultado
mapping = {
    "suspenso": 0,
    "aprobado": 1,
    "notable": 2,
    "sobresaliente": 3
}
data["final_exam_score"] = data["final_exam_score"].map(mapping)

num_cols = [
    "age",
    "weeks_in_course",
    "hours_spent_learning_per_week",
    "practice_problems_solved",
    "projects_completed",
    "tutorial_videos_watched",
    "debugging_sessions_per_week",
    "self_reported_confidence_python",
]

data[num_cols] = sklearn.preprocessing.StandardScaler().fit_transform(data[num_cols])

"""
	Aquí saco ficheros para buscar correlaciones entre distintas columnas y refinar
	más el modelo y ver que todo es correcto.
"""
data.corr().round(2).to_csv("correlacion.csv")
data.to_csv("data_cleaned.csv", index=False)


"""
	Ahora vamos a entrenar nuestros modelos:
		- SVC
		- knn

	Primero dividimos los datos en datos de entrenamiento y test.

	Luego entrenamos con los datos de entrenamiento y comprobamos
	nuestras métricas en los datos de test.
"""

# Etiquetas
X = data.drop(columns=["final_exam_score"])
# Columna a predecir
y = data["final_exam_score"]

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.3, random_state=42)

svc = sklearn.svm.SVC(random_state=42, probability=True)
svc.fit(X_train, y_train)

accuracy = svc.score(X_test, y_test)
print("Accuracy SVC:", accuracy)

# Se deberá utilizar como variable de salida al menos obligatoriamente la columna: 'final_exam_score
# Es conveniente probar otros clasificadores sobre otras columna