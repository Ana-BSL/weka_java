import weka.classifiers.bayes.NaiveBayes;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.core.Instance;
import weka.core.converters.ConverterUtils.DataSource;

public class App {

    public static void main(String[] args) throws Exception {

        DataSource ds = new DataSource("src/comorbidades.arff"); // acesso a base de dados
        Instances ins = ds.getDataSet(); // todas as instância da base de dados
        System.out.println(ins.toString()); // todos os dados carregados

        ins.setClassIndex(4); // qual atributo que se quer fazer a previsão

        NaiveBayes nb = new NaiveBayes(); // algoritmo de classificação que vai ser utilizado
        nb.buildClassifier(ins); // construindo o classificador, criando o classificador

        Instance ins2 = new DenseInstance(9); // construtor, que recebe o número de atributos
        ins2.setDataset(ins);// previsão sobre algum novo usuário
        ins2.setValue(0, 63);
        ins2.setValue(1, "sim");
        ins2.setValue(2, "nao");
        ins2.setValue(3, "nao");
        ins2.setValue(5, "sim");
        ins2.setValue(6, "nao");
        ins2.setValue(7, "nao");
        ins2.setValue(8, "sim");

        double probabilidade[] = nb.distributionForInstance(ins2); // previsão da porcentagem de uma pessoa ter
                                                                   // problemas respiratórios ou não
        /*
         * System.out.println("Sim: " + probabilidade[1]);
         * System.out.println("Não: " + probabilidade[0]);
         */
        System.out.println("Sim: " + String.format("%.4f", probabilidade[1]));
        System.out.println("Não: " + String.format("%.4f", probabilidade[0]));
    }

}
