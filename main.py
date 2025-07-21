import luigi

from piper import ProcessAllImages



def main():
    luigi.run(['ProcessAllImages', '--local-scheduler', '--workers', '8'])

if __name__ == "__main__":
    main()
