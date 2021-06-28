package knn

import (
	"knn_distribuido/utils"
	"math"
	"strconv"
	"sync"
	"time"
)

func ClassifyClass(obj utils.PersonaEncuestada, data []utils.PersonaEncuestada, K int) (class string) {
	distance_tuples := make([]utils.Tuple, 0)

	for _, persona := range data {
		distance := EuclidianDistance(obj, persona)
		distance_tuples = append(distance_tuples, utils.Tuple{Value: distance, Key: persona.Class})
	}

	ch_ordered_tuples := make(chan []utils.Tuple)
	go MergeSort(distance_tuples, ch_ordered_tuples)
	ordered_tuples := <-ch_ordered_tuples
	class = FitClass(ordered_tuples[:K])
	return class
}

func TrainingKNN(epochs int, parallel_procs int, data []utils.PersonaEncuestada) (best_K int, best_accuracy float64, time_elapsed string) {
	start := time.Now()
	i := 0
	K := 3
	accuracy_tuples := make([]utils.Tuple, 0)
	increment := parallel_procs
	var wg sync.WaitGroup

	for i <= epochs {
		if K+increment >= len(data) {
			break
		}

		wg.Add(parallel_procs)
		for j := K; j < K+increment; j++ {
			go func(K int, personas []utils.PersonaEncuestada) {
				defer wg.Done()
				classified := KNNClassification(K, personas)
				accuracy := CheckAccuracy(personas, classified)
				accuracy_tuples = append(accuracy_tuples, utils.Tuple{Value: accuracy, Key: strconv.Itoa(K)})
			}(j, data)
		}
		wg.Wait()

		K += increment
		i++
	}

	ch_ordered_acurracies := make(chan []utils.Tuple)
	go MergeSort(accuracy_tuples, ch_ordered_acurracies)
	ordered_acurracies := <-ch_ordered_acurracies
	best_Tuple := ordered_acurracies[len(ordered_acurracies)-1]
	best_K, _ = strconv.Atoi(best_Tuple.Key)
	best_accuracy = best_Tuple.Value
	time_elapsed = time.Since(start).String()
	return best_K, best_accuracy, time_elapsed
}

func KNNClassification(K int, personas []utils.PersonaEncuestada) (classified []string) {
	classified = make([]string, 0)

	for i, personaX := range personas {
		distance_tuples := make([]utils.Tuple, 0)
		for j, personaY := range personas {
			if i != j {
				distance := EuclidianDistance(personaX, personaY)
				distance_tuples = append(distance_tuples, utils.Tuple{Value: distance, Key: personaY.Class})
			}
		}
		ch_ordered_tuples := make(chan []utils.Tuple)
		go MergeSort(distance_tuples, ch_ordered_tuples)
		ordered_tuples := <-ch_ordered_tuples
		class := FitClass(ordered_tuples[:K])
		classified = append(classified, class)
	}

	return classified
}

func EuclidianDistance(persona_1 utils.PersonaEncuestada, persona_2 utils.PersonaEncuestada) (distance float64) {
	distance = 0.0

	for i := 0; i < len(persona_1.Data); i++ {
		diff := persona_1.Data[i] - persona_2.Data[i]
		pow := math.Pow(diff, 2)
		distance += pow
	}

	distance = math.Sqrt(distance)
	return distance
}

func FitClass(arr_tuples []utils.Tuple) (class string) {
	map_classes := make(map[string]int)
	map_classes["Empleado"] = 0
	map_classes["Desempleado Abierto"] = 0
	map_classes["Desempleado Oculto"] = 0

	map_min_distance := make(map[string]float64)
	map_min_distance["Empleado"] = 0.0
	map_min_distance["Desempleado Abierto"] = 0.0
	map_min_distance["Desempleado Oculto"] = 0.0

	for _, tuple := range arr_tuples {
		map_classes[tuple.Key] += 1
		if map_min_distance[tuple.Key] == 0 {
			map_min_distance[tuple.Key] = tuple.Value
		}
	}

	count := 0
	class = ""

	for key, value := range map_classes {
		if count < int(value) {
			count = int(value)
			class = key
			continue
		}
		if count == int(value) {
			if map_min_distance[key] < map_min_distance[class] {
				class = key
			}
		}
	}

	return class
}

func CheckAccuracy(personas []utils.PersonaEncuestada, classified []string) (accuracy float64) {
	correct_classes := 0

	for i := 0; i < len(personas); i++ {
		if personas[i].Class == classified[i] {
			correct_classes += 1
		}
	}

	accuracy = float64(correct_classes) / float64(len(personas))
	return accuracy
}
