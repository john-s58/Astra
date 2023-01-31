use std::time::Instant;

const G: f64 = 6.674e-11;  // gravitational constant
const DT: f64 = 1000.0; // time step

#[derive(Debug)]
pub struct Body {
    center_coords: (f64, f64, f64),
    velocity: (f64, f64, f64),
    force: (f64, f64, f64),
    mass: f64,
    radius: f64
}

impl Body {
    pub fn new(center_coords: (f64, f64, f64), mass: f64,  radius: f64) -> Body {
        Body { center_coords, velocity: (0f64, 0f64, 0f64), 
            force: (0f64, 0f64, 0f64), mass, radius}
    }
}

pub fn n_body() {
    let mut bodies: Vec<Body> = vec![];
    bodies.push(Body::new((100f64,10f64,10f64), 1000f64, 10f64));
    bodies.push(Body::new((10f64,100f64,10f64), 1000f64, 10f64));
    bodies.push(Body::new((10f64,10f64,100f64), 1000f64, 10f64));
    bodies.push(Body::new((100f64,100f64,10f64), 1000f64, 10f64));
    bodies.push(Body::new((10f64,100f64,100f64), 1000f64, 10f64));

    //initialize the vector of bodies here.

    let start_time = Instant::now();
    for _ in 0..5000 {
        for i in 0..bodies.len() {
            for j in i+1..bodies.len() {
                let (x1, y1, z1) = bodies[i].center_coords;
                let (x2, y2, z2) = bodies[j].center_coords;
                let dx = x2 - x1;
                let dy = y2 - y1;
                let dz = z2 - z1;
                let dist = (dx*dx + dy*dy + dz*dz).sqrt();
                let f = (G * bodies[i].mass * bodies[j].mass) / (dist * dist);
                let fx = f * dx / dist;
                let fy = f * dy / dist;
                let fz = f * dz / dist;
                bodies[i].force.0 += fx;
                bodies[i].force.1 += fy;
                bodies[i].force.2 += fz;
                bodies[j].force.0 -= fx;
                bodies[j].force.1 -= fy;
                bodies[j].force.2 -= fz;
            }
            
        }

        for i in 0..bodies.len() {
            let ax = bodies[i].force.0 / bodies[i].mass;
            let ay = bodies[i].force.1 / bodies[i].mass;
            let az = bodies[i].force.2 / bodies[i].mass;
            bodies[i].velocity.0 += ax * DT;
            bodies[i].velocity.1 += ay * DT;
            bodies[i].velocity.2 += az * DT;
            bodies[i].center_coords.0 += bodies[i].velocity.0 * DT;
            bodies[i].center_coords.1 += bodies[i].velocity.1 * DT;
            bodies[i].center_coords.2 += bodies[i].velocity.2 * DT;
            bodies[i].force = (0.0, 0.0, 0.0);
        }
    }
    let duration = start_time.elapsed();
    println!("Simulation took {:?}", duration);
    // Use the center_coords and radius values from the struct to display the results of the simulation
    println!("{:#?}", bodies);
}