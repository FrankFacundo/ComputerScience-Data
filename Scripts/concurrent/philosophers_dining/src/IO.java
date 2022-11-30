import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.time.LocalDateTime;

public class IO {
	private boolean used = false;
	private String nameFile = "data.txt";
	private boolean print;
	LocalDateTime DateTime = LocalDateTime.now();
	
	public IO (boolean print)
	{
		//"false" is for console and "true" is for file 
		this.print = print;
	}
	
	public synchronized void write(String text)
	{
		try {
			
			if (used == true) {
				wait();
			}
			else {
				used = true;
				try(FileWriter fw = new FileWriter(this.nameFile, true);
					    BufferedWriter bw = new BufferedWriter(fw);
					    PrintWriter out = new PrintWriter(bw))
					{
					    out.print(text + "\n");
					    
					} catch (IOException e) {
					    
					}
				
				used = false;
				notifyAll();
				
			}
			
		} catch (Exception e) {}
	}

	public void print(String text)
	{
		if (this.print == false)
			System.out.println(text);
		else
			this.write(text);
	}
	
	public boolean verifyFileExists(String name)
	{
		File file = new File("data.txt");
		return file.isFile(); 
	}
	
	public void writeHeading()
	{
		this.print("\nThis test was done on " + DateTime.toString() + "\n");
	}
	
	
}
