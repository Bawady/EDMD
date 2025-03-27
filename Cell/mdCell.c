#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "mdCell.h"
#include "mt19937ar.c"

// Size of array allocated for the event tree (currently overkill)
#define MAXEVENTS (20)

// Maximum number of cells in each direction
#define MAXCEL 100

#define MAX_FILENAME_LENGTH 1023

// Can be changed via programs args (see parse_arguments)
double maxtime = 1;             // Simulation stops at this simulation time -> same unit / non.dim. factor as simtime
int makesnapshots = 1;          // Whether to make snapshots during the run (yes = 1, no = 0)
double writeinterval = .0025;   // Time between output to screen / data file
unsigned long seed = 1;         // For seeding randomness

char input_filename[MAX_FILENAME_LENGTH + 1] = ""; // File containing an input configuration
char output_dir[MAX_FILENAME_LENGTH + 2]; // Location for the output dumps
const char *vel_output_filename = "particle_velocities.bin";
const char *pos_output_filename = "particle_positions.bin";
const char *dbg_filename = "dbg.bin";
char vel_file_path[2*MAX_FILENAME_LENGTH+1], pos_file_path[2*MAX_FILENAME_LENGTH+1], dbg_file_path[2*MAX_FILENAME_LENGTH+1];


// Config. variables for the simulation's data structures and core algorithm(s)

// Variables related to the event queueing system. These can affect efficiency.
// The system schedules only events in the current block of time with length "eventlisttime" into a sorted binary search tree.
// The rest are scheduled in unordered linked lists associated with the "numeventlists" next blocks.
// "numeventlists" is roughly equal to maxscheduletime / eventlisttime
// Any events occurring even later are put into an overflow list
// After every time block with length "eventlisttime", the set of events in the next linear list is moved into the binary search tree.
// All events in the overflow list are also rescheduled.

// After every "writeinterval", the code will output two listsizes to screen.
// The first is the average number of events in the first that gets moved into the event tree after each block.
// The second is the length of the overflow list at the last time it was looped over.
// Ideally, we set maxscheduletime large enough that the average overflow list size is negligible (i.e. <10 events)
// Also, there is some optimum value for the number of events per block (scales approximately linearly with "eventlisttime").
// I seem to get good results with an eventlisttime chosen such that there are a few hundred events per block, and dependence is pretty weak (similar performance in the range of e.g. 5 to 500 events per block...)

double maxscheduletime = 1.0;
int numeventlists;
double eventlisttimemultiplier = 1; // event list time will be this / N
double eventlisttime;

// Internal variables
double simtime = 0;
double reftime = 0;
int currentlist = 0;
int totalevents;
int N; // Number of particles

int listcounter1 = 0, listcounter2 = 0, mergecounter = 0;

event **eventlists; // Last one is overflow list

uint16_t particle_id = 0;

particle *particles;
#ifdef DIM_3D
particle *celllist[MAXCEL][MAXCEL][MAXCEL];
#else
particle *celllist[MAXCEL][MAXCEL];
#endif
event *eventlist;
event *calendar_root;
event **eventpool;
int nempty = 0;
double xsize, ysize, zsize;     // Box size
double hx, hy, hz;              // Half box size
double cxsize, cysize, czsize;  // Cell size
int cx, cy, cz;                 // Number of cells
double dvtot = 0;               // Momentum transfer (for calculating pressure)
unsigned int colcounter = 0;    // Collision counter (will probably overflow in a long run...)

double thermostatinterval = 0.01; // Time interval between applications of thermostat

int main(int argc, char **argv) {
	parse_arguments(argc, argv);
	prepare_dump_file_paths();
	fprintf(stdout, "Initializing\n");
	init();
	fprintf(stdout, "Starting simulation\n");

	while (simtime <= maxtime) {
		step();
	}
  simtime = maxtime;

	printstuff();
	advance_particles_to_sim_time();
	dump_particles();

	free(particles);
	free(eventlists);
	free(eventlist);
	free(eventpool);

	fprintf(stdout, "Simulation ended successfully\n");

	exit(EXIT_SUCCESS);
}

void prepare_dump_file_paths(){
	strncpy(vel_file_path, output_dir, MAX_FILENAME_LENGTH+2);
	strncpy(pos_file_path, output_dir, MAX_FILENAME_LENGTH+2);
	strncat(vel_file_path, vel_output_filename, MAX_FILENAME_LENGTH);
	strncat(pos_file_path, pos_output_filename, MAX_FILENAME_LENGTH);

	strncpy(dbg_file_path, output_dir, MAX_FILENAME_LENGTH+2);
	strncat(dbg_file_path, dbg_filename, MAX_FILENAME_LENGTH);
}

void parse_arguments(int argc, char **argv) {
	int opt, filename_provided = 0, out_dir_provided = 0;

	while ((opt = getopt(argc, argv, "m:d:i:f:o:s:")) != -1) {
		switch (opt) {
		case 'm':
			maxtime = atof(optarg);
			break;
		case 'd':
			makesnapshots = atoi(optarg);
			break;
		case 's':
			seed = atoi(optarg);
			break;
		case 'i':
			writeinterval = atof(optarg);
			break;
		case 'f':
			if (strlen(optarg) > MAX_FILENAME_LENGTH) {
				fprintf(stderr, "Error: Input filename exceeds %d characters.\n", MAX_FILENAME_LENGTH);
				exit(EXIT_FAILURE);
			}
			strncpy(input_filename, optarg, MAX_FILENAME_LENGTH);
			input_filename[MAX_FILENAME_LENGTH] = '\0'; // Ensure null termination
			filename_provided = 1;
			break;
		case 'o':
			if (strlen(optarg) > MAX_FILENAME_LENGTH) {
				fprintf(stderr, "Error: Output filename exceeds %d characters.\n", MAX_FILENAME_LENGTH);
				exit(EXIT_FAILURE);
			}
			// Ensure the output direcoty ends in a slash such that file names can be appended
			char *last_char = stpncpy(output_dir, optarg, MAX_FILENAME_LENGTH);
			if (*last_char != '/') {
				*last_char = '/';
			}
			output_dir[MAX_FILENAME_LENGTH+1] = '\0'; // Ensure null termination
			out_dir_provided = 1;
			break;
		default:
			fprintf(stderr, "Usage: %s -f INPUT_FILE  -o OUTPUT_DIR [-s SEED] [-m MAXTIME] [-d DO_DUMP] [-i DUMP_INTERVAL]\n", argv[0]);
			exit(EXIT_FAILURE);
		}
	}

	if (!filename_provided) {
		fprintf(stderr, "Error: Input file (-f INPUT_FILE) is required.\n");
		exit(EXIT_FAILURE);
	}

	if (!out_dir_provided) {
		fprintf(stderr, "Error: Output directory (-o OUTPUT_DIR) is required.\n");
		exit(EXIT_FAILURE);
	}
}


// Some some information about the simulation at its end
void printstuff() {
	int i;
	particle *p;
	double v2tot = 0;
	double vfilled = 0;

	for (i = 0; i < N; i++) {
		p = particles + i;
		v2tot += p->mass * v_dot(&p->vel, &p->vel);
#ifdef DIM_3D
		vfilled += p->radius * p->radius * p->radius * 8;
#else
		vfilled += p->radius * p->radius;
#endif
	}

#ifdef DIM_3D
	vfilled *= M_PI / 6.0;
	double volume = xsize * ysize * zsize;
#else
	vfilled *= M_PI;
	double volume = xsize * ysize;
#endif
	fprintf(stdout, "Average kinetic energy: %lf\n", 0.5 * v2tot / N);
	double dens = N / volume;
	double press = -dvtot / (3.0 * volume * simtime);
	double pressid = dens;
	double presstot = press + pressid;
	fprintf(stdout, "Total time simulated  : %lf\n", simtime);
	fprintf(stdout, "Packing fraction      : %lf\n", vfilled / volume);
	fprintf(stdout, "Measured pressure     : %lf + %lf = %lf\n", press, pressid, presstot);
}


void init() {
	int i;
	fprintf(stdout, "Seed: %u\n", (int)seed);
	init_genrand(seed);

	loadparticles();
	hx = 0.5 * xsize;
	hy = 0.5 * ysize;
	hz = 0.5 * zsize; // Values used for periodic boundary conditions

	// 3D
	initeventpool();
	for (i = 0; i < N; i++) {
		particle *p = particles + i;

		p->boxes_traveled.x = 0;
		p->boxes_traveled.y = 0;
#ifdef DIM_3D
		p->boxes_traveled.z = 0;
#endif

		p->t = 0;
	}
	initcelllist();
	for (i = 0; i < N; i++) {
		particle *p = particles + i;
		p->cellcrossing = eventpool[nempty - 1]; // Connect cellcrossing event to particle
		findcrossing(p);                         // Find the next cell crossing
	}

	fprintf(stdout, "Done adding cell crossings: %d events\n", totalevents - nempty);

	findallcollisions();
	fprintf(stdout, "Done adding collisions: %d events\n", totalevents - nempty);
}


void alloc_particle_array(int n) {
	N = n; // set global particle count variable
	particles = (particle *)calloc(N, sizeof(particle));
	if (!particles) {
		fprintf(stderr, "Failed to allocate memory for particles\n");
		exit(EXIT_FAILURE);
	}
}


/******************************************************
 ** Reads a single line, skipping over lines
 ** commented out with #
 ******************************************************/
int mygetline(char *str, FILE *f) {
	int comment = 1;
	while (comment) {
		if (!fgets(str, 255, f))
			return -1;
		if (str[0] != '#')
			comment = 0;
	}
	return 0;
}


/**************************************************
 ** Read particles from file
 ** First line: Number of particles
 ** Second line: box size along all three axes
 ** Rest: particle data
 ** Each particle line consists of :
 ** - A character indicating type (a = 0, b = 1, etc.)
 ** - 3 coordinates (x, y, z)
 ** - The radius
 **************************************************/
void loadparticles() {
	char tmp;
	int i, particle_cnt;
	particle *p;
	char buffer[1024];

	FILE *file;
	file = fopen(input_filename, "r");
	if (!file) {
		fprintf(stderr, "File %s not found!\n", input_filename);
		exit(EXIT_FAILURE);
	}
	mygetline(buffer, file);
	int ftmp = sscanf(buffer, "%d", &particle_cnt);
	if (ftmp != 1) {
		fprintf(stderr, "Read error (n or box)\n");
		exit(EXIT_FAILURE);
	}
	mygetline(buffer, file);
#ifdef DIM_3D
	ftmp = sscanf(buffer, "%lf %lf %lf\n", &xsize, &ysize, &zsize);
	if (ftmp != 3) {
#else
	ftmp = sscanf(buffer, "%lf %lf\n", &xsize, &ysize);
	if (ftmp != 2) {
#endif
		fprintf(stderr, "Read error (n or box)\n");
		exit(EXIT_FAILURE);
	}

	alloc_particle_array(particle_cnt);

	fprintf(stdout, "Placing particles\n");
	double vfilled = 0;
	for (i = 0; i < N; i++) {
		p = &(particles[i]);
		mygetline(buffer, file);
#ifdef DIM_3D
		ftmp = sscanf(buffer, "%c %lf %lf %lf %lf %lf %lf %lf\n", &tmp, &(p->pos.x), &(p->pos.y), &(p->pos.z), &(p->vel.x), &(p->vel.y), &(p->vel.z), &(p->radius));
#else
		ftmp = sscanf(buffer, "%c %lf %lf %lf %lf %lf\n", &tmp, &(p->pos.x), &(p->pos.y), &(p->vel.x), &(p->vel.y), &(p->radius));
#endif
		apply_periodic_boundaries(p);
#ifdef DIM_3D
		if (ftmp != 8) {
#else
		if (ftmp != 6) {
#endif
			fprintf(stderr, "Read error (particle) %d \n String: %s\n", ftmp, buffer);
			exit(EXIT_FAILURE);
		}
		p->type = tmp - 'a';
		p->mass = 1;
		vfilled += 8 * p->radius * p->radius * p->radius;
		p->cellcrossing = NULL;
		p->id = particle_id++;
	}
	fclose(file);

	// 3D
	fprintf(stdout, "Packing fraction: %lf\n", M_PI / (6.0 * xsize * ysize * zsize) * vfilled);
}


/**************************************************
 ** Update particle to the current time
 **************************************************/
void move(particle *p1) {
	// 3D
	double dt = simtime - p1->t;
	p1->t = simtime;
	f_vector tmp = v_times_scalar(&p1->vel, dt);
	v_plus(&p1->pos, &tmp);
}


/**************************************************
 ** Initialize the cell list
 ** Cell size should be at least:
 **    shellsize * [diameter of largest sphere]
 ** Can use larger cells at low densities
 **************************************************/
void initcelllist() {
	int i, j, k;
	// Number of cells in each dimension
	// OQ: The 8 * N threshold seems like some heuristic -> where from?
	cx = (int)(xsize - 0.0001);
	cy = (int)(ysize - 0.0001);
#ifdef DIM_3D
	cz = (int) zsize == 1 ? 1 : (int)(zsize - 0.0001);
	while (cx * cy * cz > 8 * N) {
#else
	while (cx * cy > 8 * N) {
#endif
		cx *= 0.9;
		cy *= 0.9;
		if (cz != 1) cz *= 0.9; // in the 2D case this is simply unused
	}

	fprintf(stdout, "Cells (X, Y, Z): (%d, %d, %d), allocating cell list memory.\n", cx, cy, cz);

	int hitmax = 0;
	if (cx > MAXCEL) {
		cx = MAXCEL;
		hitmax = 1;
	}
	if (cy > MAXCEL) {
		cy = MAXCEL;
		hitmax = 1;
	}
	if (cz > MAXCEL) {
		cz = MAXCEL;
		hitmax = 1;
	}
	if (hitmax)
		fprintf(stdout, "Maximum cellsize reduced. Consider increasing MAXCEL. New values:  %d, %d, %d\n", cx, cy, cz);

	// Size of cell in each direction
	cxsize = xsize / cx;
	cysize = ysize / cy;
	czsize = zsize / cz;

	// Clear celllist
	for (i = 0; i < cx; i++)
		for (j = 0; j < cy; j++)
#ifdef DIM_3D
			for (k = 0; k < cz; k++) {
				celllist[i][j][k] = NULL;
			}
#else
			celllist[i][j] = NULL;
#endif

	for (i = 0; i < N; i++)
		addtocelllist(particles + i);
}


/**************************************************
 ** Add particle to the cell list.
 ** Note that each cell is implemented as a doubly linked list.
 **************************************************/
void addtocelllist(particle *p) {
	p->cell.x = p->pos.x / cxsize; // Find particle's cell
	p->cell.y = p->pos.y / cysize;
#ifdef DIM_3D
	p->cell.z = p->pos.z / czsize;
#endif

	// Add particle to celllist
#ifdef DIM_3D
	p->next = celllist[p->cell.x][p->cell.y][p->cell.z];
#else
	p->next = celllist[p->cell.x][p->cell.y]; // Add particle to celllist
#endif

	if (p->next)
		p->next->prev = p; // Link up list
	p->prev = NULL;
	p->nearboxedge = (p->cell.x == 0 || p->cell.y == 0 || p->cell.x == cx - 1 || p->cell.y == cy - 1);
#ifdef DIM_3D
	p->nearboxedge |= (p->cell.z == 0 ||  p->cell.z == cz - 1);
	celllist[p->cell.x][p->cell.y][p->cell.z] = p;
#else
	celllist[p->cell.x][p->cell.y] = p;
#endif
}


void removefromcelllist(particle *p1) {
	if (p1->prev)
		p1->prev->next = p1->next; // Remove particle from celllist
	else
#ifdef DIM_3D
		celllist[p1->cell.x][p1->cell.y][p1->cell.z] = p1->next;
#else
		celllist[p1->cell.x][p1->cell.y] = p1->next;
#endif
	if (p1->next)
		p1->next->prev = p1->prev;
}


/**************************************************
 ** Process a single event
 **************************************************/
void step() {
	event *ev;
	ev = calendar_root->right;
	while (ev == NULL) // No more future events?
	{
		addnexteventlist(); // add next event bucket to the tree
		ev = calendar_root->right;
	}

	while (ev->left)
		ev = ev->left; // Find first event

	simtime = ev->eventtime;
	switch (ev->eventtype) {
	case COLLISION:
		collision(ev);
		break;
	case WRITE:
		removeevent(ev);
		dump_particles();

//		for (int i = 0; i < N; i++) {
//			particle *p = &(particles[i]);
//			fprintf(stdout, "\t%d %f %f %f %f %d %d %ld\n", p->id, p->pos.x, p->pos.y, p->vel.x, p->vel.y, p->cell.x, p->cell.y, (void*) &(*p));
//		}
		break;
	default:
		cellcross(ev);
	}
}


/**************************************************
 ** Detect the next collision for two particles
 ** Note that p1 is always up to date in
 ** findcollision
 **************************************************/
void findcollision(particle *p1, particle *p2) {
	double dt2 = simtime - p2->t;
	// relative distance at current time (pos1 - (pos2 + vel2 * dt2))
	f_vector tmp = v_times_scalar(&p2->vel, dt2);
  tmp = v_addition(&p2->pos, &tmp);
	f_vector dr = v_subtraction(&p1->pos, &tmp);

//	fprintf(stdout, "\tPositions are (%f, %f), (%f, %f)\n", p1->pos.x, p1->pos.y, p2->pos.x, p2->pos.y);
//	fprintf(stdout, "\tVelocities are (%f, %f), (%f, %f)\n", p1->vel.x, p1->vel.y, p2->vel.x, p2->vel.y);

	if (p1->nearboxedge) {
		if (dr.x > hx)
			dr.x -= xsize;
		else if (dr.x < -hx)
			dr.x += xsize; // periodic boundaries
		if (dr.y > hy)
			dr.y -= ysize;
		else if (dr.y < -hy)
			dr.y += ysize;
#ifdef DIM_3D
		if (dr.z > hz)
			dr.z -= zsize;
		else if (dr.z < -hz)
			dr.z += zsize;
#endif
	}

	f_vector dvel = v_subtraction(&p1->vel, &p2->vel); // relative velocity
	double b = v_dot(&dr, &dvel);
	if (b > 0) {
//		fprintf(stdout, "\tno collisions as b <= 0\n");
		return;
	}

	double dvel2 = v_dot(&dvel, &dvel);
	double dr2 = v_dot(&dr, &dr);
	double md = p1->radius + p2->radius;

	double discriminant = b * b - dvel2 * (dr2 - md * md);
	if (discriminant < 0){
//		fprintf(stdout, "\tNo collision as discriminant < 0\n");
		return;
	}
	double t = simtime + (-b - sqrt(discriminant)) / dvel2;
//	fprintf(stdout, "\tSchedule collision event for time %f\n", t);
	createevent(t, p1, p2, COLLISION);
}


/**************************************************
 ** Find all collisions for particle p1.
 ** The particle 'notthis' isn't checked.
 **************************************************/
void findcollisions(particle *p1, particle *notthis) // All collisions of particle p1
{
	int dx, dy, dz, ccx, ccy, ccz;
	i_vector cell = p1->cell;
	particle *p2;

	for (dx = -1; dx < 2; dx++) {
		ccx = cell.x + dx;
		if (dx < 0 && ccx < 0)
			ccx += cx;
		else if (dx > 0 && ccx >= cx)
			ccx -= cx;
		{
			for (dy = -1; dy < 2; dy++) {
				ccy = cell.y + dy;
				if (dy < 0 && ccy < 0)
					ccy += cy;
				else if (dy > 0 && ccy >= cy)
					ccy -= cy;
#ifdef DIM_3D
				for (dz = -1; dz < 2; dz++) {
					ccz = cell.z + dz;
					if (dz < 0 && ccz < 0)
						ccz += cz;
					else if (dz > 0 && ccz >= cz)
						ccz -= cz;
					for (p2 = celllist[ccx][ccy][ccz]; p2; p2 = p2->next)
						if (p2 != p1 && p2 != notthis)
							findcollision(p1, p2);
				}
#else
					for (p2 = celllist[ccx][ccy]; p2; p2 = p2->next){
						if (p2 != p1 && p2 != notthis)
							findcollision(p1, p2);
					}
#endif
			}
		}
	}
}


/**************************************************
 ** All collisions of all particle pairs
 **************************************************/
void findallcollisions() // All collisions of all particle pairs
{
	int i, dx, dy, cellx, celly;
#ifdef DIM_3D
	int dz, cellz;
#endif
	particle *p1, *p2;

//	fprintf(stdout, "Initialize particle collision events\n");

	for (i = 0; i < N; i++) {
		// plus amount of cells to mitigate obtaining a negative cell index at the nested loop below -> % in index calculation makes this a periodic-boundary behavior
		cellx = particles[i].cell.x + cx;
		celly = particles[i].cell.y + cy;
#ifdef DIM_3D
		cellz = particles[i].cell.z + cz;
#endif
		p1 = &(particles[i]);

		for (dx = cellx - 1; dx <= cellx + 1; dx++){
			for (dy = celly - 1; dy <= celly + 1; dy++){
#ifdef DIM_3D
				for (dz = cellz - 1; dz <= cellz + 1; dz++) {
					p2 = celllist[dx % cx][dy % cy][dz % cz];
#else
					p2 = celllist[dx % cx][dy % cy];
#endif
					while (p2) {
						// compare memory addresses to ensure pair of particles are only checked once per step
						if (p2 > p1) {
//							fprintf(stdout, "\tCheck if %d collides with %d\n", p1->id, p2->id);
							findcollision(p1, p2);
						}
						p2 = p2->next;
					}
#ifdef DIM_3D
				}
#endif
			}
		}
	}
}


/**************************************************
 ** Find all collisions for particle p1 after a
 ** cellcrossing.
 ** This could find events that have already been
 ** scheduled, but they will be deleted as soon
 ** as one of the pair is processed.
 **************************************************/
void findcollisioncell(particle *p1, int type) {
	int dx, dy, ccx, ccy;
	i_vector cell = p1->cell;
	particle *p2;

	int xmin = cell.x - 1;
	int xmax = cell.x + 1;
	int ymin = cell.y - 1;
	int ymax = cell.y + 1;
#ifdef DIM_3D
	int dz, ccz;
	int zmin = cell.z - 1;
	int zmax = cell.z + 1;
#endif
	switch (type) {
	case -1: // Negative x-direction
		xmax = xmin;
		break;
	case 1: // Positive x-direction
		xmin = xmax;
		break;
	case -2: // Negative y-direction
		ymax = ymin;
		break;
	case 2: // Positive y-direction
		ymin = ymax;
		break;
#ifdef DIM_3D
	case -3: // Negative z-direction
		zmax = zmin;
		break;
	case 3: // Positive z-direction
		zmin = zmax;
		break;
#endif
	}

	for (dx = xmin; dx <= xmax; dx++) {
		if (dx < 0)
			ccx = dx + cx;
		else if (dx >= cx)
			ccx = dx - cx;
		else
			ccx = dx;
		for (dy = ymin; dy <= ymax; dy++) {
			if (dy < 0)
				ccy = dy + cy;
			else if (dy >= cy)
				ccy = dy - cy;
			else
				ccy = dy;
#ifdef DIM_3D
			for (dz = zmin; dz <= zmax; dz++) {
				if (dz < 0)
					ccz = dz + cz;
				else if (dz >= cz)
					ccz = dz - cz;
				else
					ccz = dz;
				p2 = celllist[ccx][ccy][ccz];
#else
				p2 = celllist[ccx][ccy];
#endif
				while (p2) {
					if (p2 != p1)
						findcollision(p1, p2);
					p2 = p2->next;
				}
#ifdef DIM_3D
			}
#endif
		}
	}
}


/**************************************************
 ** Process a single collision event
 **************************************************/
void collision(event *ev) {
	particle *p1 = ev->p1;
	particle *p2 = ev->p2;

	double m1 = p1->mass, r1 = p1->radius;
	double m2 = p2->mass, r2 = p2->radius;

	move(p1);
	move(p2);

	double r = r1 + r2;
	f_vector dr = v_subtraction(&p1->pos, &p2->pos);

	if (p1->nearboxedge) {
		if (dr.x > hx)
			dr.x -= xsize;
		else if (dr.x < -hx)
			dr.x += xsize; // periodic boundaries
		if (dr.y > hy)
			dr.y -= ysize;
		else if (dr.y < -hy)
			dr.y += ysize;
#ifdef DIM_3D
		if (dr.z > hz)
			dr.z -= zsize;
		else if (dr.z < -hz)
			dr.z += zsize;
#endif
	}

	double rinv = 1.0 / r;
	v_scalar_product(&dr, rinv);

	f_vector dvel = v_subtraction(&p1->vel, &p2->vel); // relative velocity

	double b = v_dot(&dr, &dvel); // dr.dv
	b *= 2.0 / (m1 + m2);
	double dv1 = b * m2, dv2 = b * m1;

	f_vector tmp = v_times_scalar(&dr, dv1);
	v_minus(&p1->vel, &tmp);
	tmp = v_times_scalar(&dr, dv2);
	v_plus(&p2->vel, &tmp);

	dvtot += dv1 * m1 * r;
	colcounter++;

	event *del = p1->cellcrossing; // Delete all old events for these particles
	while (del->nextp1 != del)
		removeevent(del->nextp1); // This includes 'ev' itself
	while (del->nextp2 != del)
		removeevent(del->nextp2);
	removeevent(del);
	findcrossing(p1); // Find new cellcrossings
	findcollisions(p1, p2);

	del = p2->cellcrossing;
	while (del->nextp1 != del)
		removeevent(del->nextp1);
	while (del->nextp2 != del)
		removeevent(del->nextp2);
	removeevent(del);
	findcrossing(p2);
	findcollisions(p2, p1);
}


/**************************************************
 ** Find the next cellcrossing for a particle
 **************************************************/
void findcrossing(particle *part) {
	double t, tmin;
	int type; // Type 1,2,3 refers to x,y,z direction, -1,-2,-3 to negative x,y,z direction
	double vx = part->vel.x;
	double vy = part->vel.y;

	if (vx < 0) {
		tmin = (part->cell.x * cxsize - part->pos.x) / vx;
		type = XMINUS;
	} else {
		tmin = ((part->cell.x + 1) * cxsize - part->pos.x) / vx;
		type = XPLUS;
	}
	if (vy < 0) {
		t = (part->cell.y * cysize - part->pos.y);
		if (t > tmin * vy) {
			type = YMINUS;
			tmin = t / vy;
		}
	} else {
		t = ((part->cell.y + 1) * cysize - part->pos.y);
		if (t < tmin * vy) {
			type = YPLUS;
			tmin = t / vy;
		}
	}
#ifdef DIM_3D
	double vz = part->vel.z;
	if (vz < 0) {
		t = (part->cell.z * czsize - part->pos.z);
		if (t > tmin * vz) {
			type = ZMINUS;
			tmin = t / vz;
		}
	} else {
		t = ((part->cell.z + 1) * czsize - part->pos.z);
		if (t < tmin * vz) {
			type = ZPLUS;
			tmin = t / vz;
		}
	}
#endif
	createevent(part->t + tmin, part, part, type);
}

/**************************************************
 **                  CELLCROSS
 ** Process a single cellcrossing event
 **************************************************/
void cellcross(event *ev) {
	particle *part = ev->p1;
	int type = ev->eventtype;
	double pt = simtime - part->t;

	f_vector tmp = v_times_scalar(&part->vel, pt);
	v_plus(&part->pos, &tmp);
	part->t = simtime;

 // Remove particle from cell's linked list and change pointers of predecessor and successor (if any)
	if (part->prev) {
		part->prev->next = part->next;
	}
	else {
#ifdef DIM_3D
		celllist[part->cell.x][part->cell.y][part->cell.z] = part->next;
#else
		celllist[part->cell.x][part->cell.y] = part->next;
#endif
	}
	if (part->next) {
		part->next->prev = part->prev;
	}

	switch (type) {
	case -1: //-x
		if (part->cell.x == 0) {
			part->cell.x = cx - 1;
			part->pos.x += xsize;
			part->boxes_traveled.x--;
		} else
			part->cell.x--;
		break;
	case 1: //+x
		part->cell.x++;
		if (part->cell.x == cx) {
			part->cell.x = 0;
			part->pos.x -= xsize;
			part->boxes_traveled.x++;
		}
		break;
	case -2: //-y
		if (part->cell.y == 0) {
			part->cell.y = cy - 1;
			part->pos.y += ysize;
			part->boxes_traveled.y--;
		} else
			part->cell.y--;
		break;
	case 2: //+y
		part->cell.y++;
		if (part->cell.y == cy) {
			part->cell.y = 0;
			part->pos.y -= ysize;
			part->boxes_traveled.y++;
		}
		break;
#ifdef DIM_3D
	case -3: //-z
		if (part->cell.z == 0) {
			part->cell.z = cz - 1;
			part->pos.z += zsize;
			part->boxes_traveled.z--;
		} else
			part->cell.z--;
		break;
	case 3: //+z
		part->cell.z++;
		if (part->cell.z == cz) {
			part->cell.z = 0;
			part->pos.z -= zsize;
			part->boxes_traveled.z++;
		}
		break;
#endif
	}

	part->prev = NULL; // Add particle to celllist
#ifdef DIM_3D
	part->next = celllist[part->cell.x][part->cell.y][part->cell.z];
	celllist[part->cell.x][part->cell.y][part->cell.z] = part;
#else
	part->next = celllist[part->cell.x][part->cell.y];
	celllist[part->cell.x][part->cell.y] = part;
#endif
	if (part->next)
		part->next->prev = part;
	part->nearboxedge = (part->cell.x == 0 || part->cell.y == 0 || part->cell.x == cx - 1 || part->cell.y == cy - 1);
#ifdef DIM_3D
	part->nearboxedge |= part->cell.z == 0 || part->cell.z == cz - 1;
#endif

	removeevent(ev);                // Note that the next event added after this HAS to be the new cellcrossing
	findcrossing(part);             // Find next crossing
	findcollisioncell(part, type);  // Find collisions with particles in new neighbouring cells
}

/**************************************************
 **                 INITEVENTPOOL
 **************************************************/
void initeventpool() {
	eventlisttime = eventlisttimemultiplier / N;           // Delta t
	numeventlists = ceil(maxscheduletime / eventlisttime); // t_max
	maxscheduletime = numeventlists * eventlisttime;
	fprintf(stdout, "number of event lists: %d\n", numeventlists);

	// Amount of regular event buckets + overflow bucket
	eventlists = (event **)calloc(numeventlists + 1, sizeof(event *));
	if (!eventlists) {
		fprintf(stderr, "Failed to allocate memory for eventlists\n");
		exit(EXIT_FAILURE);
	}

	totalevents = N * MAXEVENTS;
	eventlist = (event *)malloc(totalevents * sizeof(event));
	eventpool = (event **)malloc(totalevents * sizeof(event *));
	int i;
	event *e;
	for (i = 0; i < totalevents; i++) // Clear eventpool
	{
		e = &(eventlist[totalevents - i - 1]); // Fill in list of free events
		eventpool[i] = e;
		eventpool[i]->left = NULL;
		eventpool[i]->right = NULL;
		e->nextp1 = e;
		e->prevp1 = e; // Initialize circular lists. Only useful for the cell crossings.
		e->nextp2 = e;
		e->prevp2 = e;
		nempty++; // nempty keeps track of the number of free events
	}
	calendar_root = eventpool[--nempty];        // Create root event
	calendar_root->eventtime = -99999999999.99; // Root event is empty, but makes sure every other event has a parent
	calendar_root->eventtype = ROOT;            // This makes sure we don't have to keep checking this when adding/removing events
	calendar_root->parent = NULL;

	event *writeevent = eventpool[--nempty]; // Setup write event
	writeevent->eventtime = 0;
	writeevent->eventtype = WRITE;
	calendar_root->right = writeevent;
	writeevent->parent = calendar_root;
	fprintf(stdout, "Event tree initialized: %d events\n", totalevents - nempty);
}

/**************************************************
 **                  ADDEVENTTOTREE
 ** Add event into binary tree
 **************************************************/
void addeventtotree(event *newevent) {
	double time = newevent->eventtime;
	event *loc = calendar_root;
	int busy = 1;
	while (busy) // Find location to add event into tree (loc)
	{
		if (time < loc->eventtime) // Go left
		{
			if (loc->left)
				loc = loc->left;
			else {
				loc->left = newevent;
				busy = 0;
			}
		} else // Go right
		{
			if (loc->right)
				loc = loc->right;
			else {
				loc->right = newevent;
				busy = 0;
			}
		}
	}
	newevent->parent = loc;
	newevent->left = NULL;
	newevent->right = NULL;
}

/**************************************************
 **                  ADDEVENT
 ** Add event to event calendar
 **************************************************/
void addevent(event *newevent) {
	double dt = newevent->eventtime - reftime;

	if (dt < eventlisttime) // Event in near future if dt within Delta t -> Put it in the tree insteaf of a bucket
	{
		newevent->queue = currentlist;
		addeventtotree(newevent);
	} else // Put it in one of the event lists
	{
		int list_id;
		if (dt >= numeventlists * eventlisttime)
			list_id = numeventlists; // This also handles int overflow when calculating list_id
		else {
			list_id = currentlist + dt / eventlisttime;
			if (list_id >= numeventlists) {
				list_id -= numeventlists;
			}
		}

		newevent->queue = list_id;
		newevent->right = eventlists[list_id]; // Add to linear list
		newevent->left = NULL;
		if (newevent->right)
			newevent->right->left = newevent;
		eventlists[list_id] = newevent;
	}
}
/**************************************************
 **                  CREATEEVENT
 ** Create a new event at the provided time,
 ** involving particles p1, and p2,
 ** and with the provided event type.
 **************************************************/
void createevent(double time, particle *p1, particle *p2, int type) {
	event *newevent = eventpool[--nempty]; // Pick first unused event
	newevent->eventtime = time;
	newevent->p1 = p1;
	newevent->eventtype = type;

//	fprintf(stdout, "Create event of type %s at time %f and position %d\n", event_type_to_str(type), time, nempty);

	if (type == COLLISION)
	{
		newevent->p2 = p2;
		event *cc = newevent->p1->cellcrossing; // Use cellcrossing event with each particle
		newevent->nextp1 = cc->nextp1;          //... to link this event into their lists
		cc->nextp1 = newevent;                  // cellcrossing -> this event -> other ...
		newevent->prevp1 = cc;
		newevent->nextp1->prevp1 = newevent;

		cc = newevent->p2->cellcrossing;
		newevent->nextp2 = cc->nextp2;
		cc->nextp2 = newevent;
		newevent->prevp2 = cc;
		newevent->nextp2->prevp2 = newevent;
	}

	addevent(newevent);
}

/**************************************************
 **                     ADDNEXTEVENTLIST
 ** Sort all events from the first event list
 ** into the binary tree.
 **************************************************/
void addnexteventlist() {
	currentlist++;
	reftime += eventlisttime;
	if (currentlist == numeventlists) // End of array of event lists?
	{
		currentlist = 0; // Start at the beginning again
										 // Process overflow queue
		event *ev = eventlists[numeventlists];
		eventlists[numeventlists] = NULL;
		listcounter2 = 0;
		while (ev) {
			event *nextev = ev->right;
			addevent(ev);
			ev = nextev;
			listcounter2 += 1; // Count how many events there were in overflow
		}
	}

	event *ev = eventlists[currentlist];
	while (ev) {
		event *nextev = ev->right;
		addeventtotree(ev);
		ev = nextev;
		listcounter1++; // Count how many events there were in event list
	}
	eventlists[currentlist] = NULL;
	mergecounter++;
}

/**************************************************
 **                  REMOVEEVENT
 ** Remove an event from the event calendar
 **************************************************/
void removeevent(event *oldevent) {

	if (oldevent->eventtype == COLLISION) // Update doubly-linked lists of events involving particles in case of collision
	{
		oldevent->nextp1->prevp1 = oldevent->prevp1;
		oldevent->prevp1->nextp1 = oldevent->nextp1;
		oldevent->nextp2->prevp2 = oldevent->prevp2;
		oldevent->prevp2->nextp2 = oldevent->nextp2;
	} else if (oldevent->eventtype == 8) {
		oldevent->nextp1->prevp1 = oldevent->prevp1;
		oldevent->prevp1->nextp1 = oldevent->nextp1;
	}

	if (oldevent->queue != currentlist) // Not in the binary tree (i.e., in some future event bucket)
	{
		if (oldevent->right)
			oldevent->right->left = oldevent->left;
		if (oldevent->left)
			oldevent->left->right = oldevent->right;
		else {
			eventlists[oldevent->queue] = oldevent->right;
		}
		eventpool[nempty++] = oldevent; // Put the removed event back in the event pool.
		return;
	}

	event *parent = oldevent->parent;
	event *node; // This node will be attached to parent in the end

	if (oldevent->left == NULL) // Only one child: easy to delete
	{
		node = oldevent->right; // Child2 is attached to parent
		if (node) {
			node->parent = parent;
		}
	} else if (oldevent->right == NULL) // Only one child again
	{
		node = oldevent->left; // Child1 is attached to parent
		node->parent = parent;
	} else // Node to delete has 2 children
	{			 // In this case: a) Find first node after oldevent     (This node will have no left)
		//               b) Remove this node from the tree     (Attach node->right to node->parent)
		//               c) Put this node in place of oldevent (Oldevent's children are adopted by node)
		node = oldevent->right;
		while (node->left)
			node = node->left; // Find first node of right tree of descendants of oldevent
		event *pnode = node->parent;
		if (pnode != oldevent)			 // node is not a child of oldevent
		{														 // Both of oldevent's children should be connected to node
			pnode->left = node->right; // Remove node from right tree
			if (node->right)
				node->right->parent = pnode;
			oldevent->left->parent = node;
			node->left = oldevent->left;
			oldevent->right->parent = node;
			node->right = oldevent->right;
		} else // This means node == oldevent->right
		{			 // Only left has to be attached to node
			oldevent->left->parent = node;
			node->left = oldevent->left;
		}
		node->parent = parent;
	}
	if (parent->left == oldevent)
		parent->left = node;
	else
		parent->right = node;
	eventpool[nempty++] = oldevent; // Put the removed event back in the event pool.
}


void advance_particles_to_sim_time() {
	int i;
	particle *p;
	for (i = 0; i < N; i++) {
		p = &(particles[i]);
		double dt = simtime - p->t;
		f_vector tmp = v_times_scalar(&p->vel, dt);
		v_plus(&p->pos, &tmp);
		p->t = simtime;
	}
}


void dump_particles() {
	static int counter = 0;
	static int first = 1;
	static double dvtotlast = 0;
	static double timelast = 0;
	int i;
	particle *p;
	FILE *vel_file, *pos_file, *dbg_file;

	double en = 0;
	for (i = 0; i < N; i++) {
		p = particles + i;
		en += p->mass * v_dot(&p->vel, &p->vel);
	}
	double temperature = 0.5 * en / (float)N / 1.5;

	double volume = xsize * ysize;
#ifdef DIM_3D
	volume *= zsize;
#endif
	double pressid = (double)N / volume;
	double pressnow = -(dvtot - dvtotlast) / (3.0 * volume * (simtime - timelast));
	pressnow += pressid;
	dvtotlast = dvtot;
	timelast = simtime;
	if (colcounter == 0)
		pressnow = 0;

	double listsize1 = (double)listcounter1 / mergecounter; // Average number of events in the first event list
	int listsize2 = listcounter2;                           // Number of events in overflow list during last rescheduling (0 if not recently rescheduled)
	if (mergecounter == 0)
		listsize1 = 0;
	listcounter1 = listcounter2 = mergecounter = 0;

	fprintf(stdout, "Simtime: %lf, Collisions: %u, Press: %lf, T: %lf, Listsizes: (%lf, %d)\n",
	        simtime, colcounter, pressnow, temperature, listsize1, listsize2);
	if (makesnapshots) {
		if (first) {
			first = 0;
			pos_file = fopen(pos_file_path, "wb");
			vel_file = fopen(vel_file_path, "wb");
			dbg_file = fopen(dbg_file_path, "wb");
		}
		else {
			pos_file = fopen(pos_file_path, "ab");
			vel_file = fopen(vel_file_path, "ab");
			dbg_file = fopen(dbg_file_path, "ab");
		}
 //		fprintf(file, "%d\n%.12lf %.12lf %.12lf\n%.12lf\n", (int)N, xsize, ysize, zsize, temperature);
		for (i = 0; i < N; i++) {
			p = &(particles[i]);
			move(p);

#ifdef DIM_3D
			double data[] = {p->vel.x, p->vel.y, p->vel.z};
			fwrite(data, sizeof(double), 3, vel_file);
			data[0] = p->pos.x;
			data[1] = p->pos.y;
			data[2] = p->pos.z;
			fwrite(data, sizeof(double), 3, pos_file);
#else
			double data[] = {p->vel.x, p->vel.y};
			fwrite(data, sizeof(double), 2, vel_file);
			data[0] = p->pos.x;
			data[1] = p->pos.y;
			fwrite(data, sizeof(double), 2, pos_file);
			uint16_t id[] = {p->id};
			fwrite(id, sizeof(uint16_t), 1, dbg_file);
#endif
		}
		fclose(vel_file);
		fclose(pos_file);
		fclose(dbg_file);
	}

	counter++;

	// Schedule next write event
	createevent(simtime + writeinterval, NULL, NULL, 100); // Add next write interval
}

/**************************************************
 ** Apply periodic boundaries
 ** Just for initialization
 **************************************************/
void apply_periodic_boundaries(particle *p) {
	p->pos.x -= xsize * floor(p->pos.x / xsize);
	p->pos.y -= ysize * floor(p->pos.y / ysize);
#ifdef DIM_3D
	p->pos.z -= zsize * floor(p->pos.z / zsize);
#endif
}


/**************************************************
 **                    RANDOM_GAUSSIAN
 ** Generates a random number from a
 ** Gaussian distribution (Box–Muller)
 **************************************************/
double random_gaussian() {
	static int have_deviate = 0;
	static double u1, u2;
	double x1, x2, w;

	if (have_deviate) {
		have_deviate = 0;
		return u2;
	} else {
		do {
			x1 = 2.0 * genrand_real2() - 1.0;
			x2 = 2.0 * genrand_real2() - 1.0;
			w = x1 * x1 + x2 * x2;
		} while (w >= 1.0);
		w = sqrt((-2.0 * log(w)) / w);
		u1 = x1 * w;
		u2 = x2 * w;
		have_deviate = 1;
		return u1;
	}
}
