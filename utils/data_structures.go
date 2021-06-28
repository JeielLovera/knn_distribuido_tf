package utils

type PersonaEncuestada struct {
	Data  []float64
	Class string
}

type Tuple struct {
	Value float64
	Key   string
}

type TrainedData struct {
	Best_k        int
	Best_accuracy float64
}

type TrainingK struct {
	Epochs          int
	Current_K       int
	Parallel_procs  int
	Accuracy_tuples []Tuple
	Personas        []PersonaEncuestada
}
