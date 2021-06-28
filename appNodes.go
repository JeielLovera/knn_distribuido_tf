package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"knn_distribuido/knn"
	"knn_distribuido/utils"
	"math/rand"
	"net"
	"os"
	"strconv"
	"strings"
	"sync"
)

/* VARIABLES GLOBALES */
var addrs []string
var addr_node string
var arr_scores []utils.Tuple
var stop_training chan bool

const (
	register_port     = 8000
	notification_port = 8001
	service_knn_port  = 8002 // servicio de entrenamiento del KNN
	service_send_port = 8003 // servicio para enviar el KNN entrenado
)

func localAddress() string {
	ifaces, err := net.Interfaces()

	if err != nil {
		fmt.Printf("error")
	}

	for _, oiface := range ifaces {
		if strings.Contains(oiface.Name, "local") {
			addrs, err := oiface.Addrs()

			if err != nil {
				fmt.Printf("error")
				continue
			}

			for _, dir := range addrs {
				switch d := dir.(type) {
				case *net.IPNet:
					if strings.HasPrefix(d.IP.String(), "192") {
						return d.IP.String()
					}
				}
			}
		}
	}
	return "127.0.0.1"
}

/* FUNCIONES COMO SERVIDOR */
func RegisterServer() {
	hostname := fmt.Sprintf("%s:%d", addr_node, register_port)
	listen, _ := net.Listen("tcp", hostname)
	defer listen.Close()

	for {
		conn, _ := listen.Accept()
		go HandleRegister(conn)
	}
}

func HandleRegister(conn net.Conn) {
	defer conn.Close()

	bufferIn := bufio.NewReader(conn)
	ip, _ := bufferIn.ReadString('\n')
	ip = strings.TrimSpace(ip)

	bytes, _ := json.Marshal(addrs)

	fmt.Fprintf(conn, "%s\n", string(bytes))

	NotifyAllNodes(ip)

	addrs = append(addrs, ip)
	fmt.Println(addrs)
}

func NotifyAllNodes(ip string) {
	for _, addr := range addrs {
		Notify(addr, ip)
	}
}

func Notify(addr string, ip string) {
	hostremote := fmt.Sprintf("%s:%d", addr, notification_port)
	conn, _ := net.Dial("tcp", hostremote)

	defer conn.Close()

	fmt.Fprintf(conn, "%s\n", ip)
}

func ListenNotifications() {
	hostname := fmt.Sprintf("%s:%d", addr_node, notification_port)
	listen, _ := net.Listen("tcp", hostname)

	defer listen.Close()

	for {
		conn, _ := listen.Accept()
		go HandleNotification(conn)
	}
}

func HandleNotification(conn net.Conn) {
	defer conn.Close()

	bufferIn := bufio.NewReader(conn)
	ip, _ := bufferIn.ReadString('\n')
	ip = strings.TrimSpace(ip)

	addrs = append(addrs, ip)
	fmt.Println(addrs)
}

/* FUNCIONES COMO CLIENTE */
func RegisterClient(hostremote string) {
	remote_port := fmt.Sprintf("%s:%d", hostremote, register_port)
	conn, _ := net.Dial("tcp", remote_port)

	defer conn.Close()

	fmt.Fprintf(conn, "%s\n", addr_node)

	bufferIn := bufio.NewReader(conn)
	bitacora, _ := bufferIn.ReadString('\n')

	var arrtemp []string
	json.Unmarshal([]byte(bitacora), &arrtemp)

	addrs = append(arrtemp, hostremote)
	fmt.Println(addrs)
}

/* FUNCIONES ENTRENAMIENTO KNN CON P2P */
func RegisterServerHP() {
	hostname := fmt.Sprintf("%s:%d", addr_node, service_knn_port)
	listen, _ := net.Listen("tcp", hostname)

	defer listen.Close()

	for {
		conn, _ := listen.Accept()
		go HandleProcessHP(conn)
	}
}

func HandleProcessHP(conn net.Conn) {
	defer conn.Close()

	bufferIn := bufio.NewReader(conn)
	obj, _ := bufferIn.ReadString('\n')
	var training_k utils.TrainingK
	json.Unmarshal([]byte(obj), &training_k)
	fmt.Println("Numero recibido: ", training_k.Epochs)
	fmt.Println("Data entrenada recibida: ", training_k.Accuracy_tuples)

	// logica del proceso
	if training_k.Epochs == 0 {
		fmt.Println("Termino el proceso")
		arr_scores = training_k.Accuracy_tuples
		stop_training <- true
	} else {
		// entrenamiento del knn
		var wg sync.WaitGroup
		increment := training_k.Parallel_procs

		if training_k.Current_K+increment >= len(training_k.Personas) {
			arr_scores = training_k.Accuracy_tuples
			stop_training <- true
		}

		wg.Add(training_k.Parallel_procs)
		for j := training_k.Current_K; j < training_k.Current_K+increment; j++ {
			go func(K int, personas []utils.PersonaEncuestada) {
				defer wg.Done()
				classified := knn.KNNClassification(K, personas)
				accuracy := knn.CheckAccuracy(personas, classified)
				training_k.Accuracy_tuples = append(training_k.Accuracy_tuples, utils.Tuple{Value: accuracy, Key: strconv.Itoa(K)})
			}(j, training_k.Personas)
		}
		wg.Wait()

		training_k.Current_K += increment
		training_k.Epochs -= 1
		SendToNextNode(training_k)
	}
}

func SendToNextNode(training_k utils.TrainingK) {
	next_host := ""
	if training_k.Epochs != 0 {
		idx := rand.Intn(len(addrs))
		next_host = addrs[idx]
	} else {
		// en el último numero de carga regresa al nodo principal
		next_host = "192.168.0.2"
	}

	fmt.Println("")
	fmt.Printf("Enviando %d a %s\n", training_k.Epochs, next_host)

	hostremote := fmt.Sprintf("%s:%d", next_host, service_knn_port)
	conn, _ := net.Dial("tcp", hostremote)

	defer conn.Close()

	bytes, _ := json.Marshal(training_k)
	fmt.Fprintln(conn, string(bytes))
}

/* FUNCIONES DE ENVIO DE KNN ENTRENADO */
func SendTrainedData(data utils.TrainedData) {
	ip := "192.168.0.2" // ip nodo principal
	port := strconv.Itoa(service_send_port)
	hostremote := ip + ":" + port
	conn, _ := net.Dial("tcp", hostremote)

	defer conn.Close()

	bytes, _ := json.Marshal(data)
	fmt.Fprintf(conn, "%s\n", string(bytes))
}

/* MAIN */
func main() {
	stop_training = make(chan bool)
	addr_node = localAddress()
	fmt.Println("IP: ", addr_node)

	go RegisterServer()

	// servicio entrenar KNN (escucha)
	go RegisterServerHP()

	go func() {
		for {
			<-stop_training

			ch_ordered_acurracies := make(chan []utils.Tuple)
			go knn.MergeSort(arr_scores, ch_ordered_acurracies)
			ordered_acurracies := <-ch_ordered_acurracies
			best_Tuple := ordered_acurracies[len(ordered_acurracies)-1]
			best_K, _ := strconv.Atoi(best_Tuple.Key)
			best_accuracy := best_Tuple.Value
			trained_data := utils.TrainedData{Best_k: best_K, Best_accuracy: best_accuracy}
			fmt.Println("----trained data---")
			fmt.Println(trained_data)
			fmt.Println("")
			// envio de KNN entrenado
			SendTrainedData(trained_data)
			close(stop_training)
			stop_training = make(chan bool)
		}
	}()

	bufferIn := bufio.NewReader(os.Stdin)
	fmt.Printf("Ingrese ip del nodo a ingresar: ")

	hostremote, _ := bufferIn.ReadString('\n')
	hostremote = strings.TrimSpace(hostremote)

	if hostremote != "" {
		RegisterClient(hostremote)
	}

	ListenNotifications()
}
