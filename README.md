# Analysis and Simulation of Power Systems

**amps** is a lightweight and extensible circuit simulation tool written in Python.  

## Structure

          +---------+
          | Parser  |
          +---------+
               |
               v
     +-------------------+
     |   Circuit Graph   |
     +-------------------+
               |
               v
      +--------------------+
      |   Engine           |
      | - builds models    |
      | - sets up solver   |
      | - runs simulation  |
      +--------------------+
         /           \
        v             v
+----------------+   +--------+
| ElectricalModel|   | Solver |
+----------------+   +--------+
                         |
                         v
              +--------------------+
              |   Simulation Output |
              +--------------------+

## License
MIT License. See `LICENSE` file for details.