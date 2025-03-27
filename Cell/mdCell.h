

typedef enum event_type_t {COLLISION=0, ROOT=99, WRITE=100, XMINUS=-1, XPLUS=1, YMINUS=-2, YPLUS=2, ZMINUS=-3, ZPLUS=3};

//Event structure
typedef struct sevent
{
    double eventtime;
    struct sevent* left;    //Left child in tree or previous event in event list
    struct sevent* right;   //Right child in tree or next event in event list
    struct sevent* parent;  //Parent in tree
    struct sparticle* p1;   //Particles involved in the event
    struct sparticle* p2;
    struct sevent* prevp1, *nextp1; //Circular linked list for all events involving p1 as the first particle
    struct sevent* prevp2, *nextp2; //Circular linked list for all events involving p2 as the second particle
    int eventtype;
    int queue;    //Index of the event queue this event is in
} event;

typedef struct {
  double x, y;
#ifdef DIM_3D
  double z;
#endif
} f_vector;

typedef struct {
  int x, y;
#ifdef DIM_3D
  int z;
#endif
} i_vector;

static inline void v_plus(f_vector *v1, const f_vector *v2) {
    v1->x += v2->x;
    v1->y += v2->y;
#ifdef DIM_3D
    v1->z += v2->z;
#endif
}

static inline void v_minus(f_vector *v1, const f_vector *v2) {
    v1->x -= v2->x;
    v1->y -= v2->y;
#ifdef DIM_3D
    v1->z -= v2->z;
#endif
}

static inline void v_scalar_product(f_vector *v, double s) {
    v->x *= s;
    v->y *= s;
#ifdef DIM_3D
    v->z *= s;
#endif
}

static inline f_vector v_subtraction(const f_vector *v1, const f_vector *v2) {
    f_vector result;
    result.x = v1->x - v2->x;
    result.y = v1->y - v2->y;
#ifdef DIM_3D
    result.z = v1->z - v2->z;
#endif
    return result;
}

static inline f_vector v_addition(const f_vector *v1, const f_vector *v2) {
    f_vector result;
    result.x = v1->x + v2->x;
    result.y = v1->y + v2->y;
#ifdef DIM_3D
    result.z = v1->z + v2->z;
#endif
    return result;
}

static inline f_vector v_multiplication(const f_vector *v1, const f_vector *v2) {
    f_vector result;
    result.x = v1->x * v2->x;
    result.y = v1->y * v2->y;
#ifdef DIM_3D
    result.z = v1->z * v2->z;
#endif
    return result;
}

static inline f_vector v_times_scalar(const f_vector *v, double s) {
    f_vector result;
    result.x = s * v->x;
    result.y = s * v->y;
#ifdef DIM_3D
    result.z = s * v->z;
#endif
    return result;
}

static inline double v_dot(const f_vector *v1, const f_vector *v2) {
    double result;
    result = v1->x * v2->x;
    result += v1->y * v2->y;
#ifdef DIM_3D
    result += v1->z * v2->z;
#endif
    return result;
}

//Particle structure
typedef struct sparticle
{
  f_vector pos;
  f_vector vel;
  double t;                       // Last update time
  double radius;
  double mass;
  uint8_t nearboxedge;            // Is this particle in a cell near the box edge?
  i_vector cell;                  // Current cell
  i_vector boxes_traveled;        // Keeps track of dynamics across periodic boundaries
  event* cellcrossing;            // Cell crossing event
  struct sparticle* prev, *next;  // Doubly linked cell list
  uint8_t type;                   // Particle type
  uint16_t id;
} particle;

int main(int, char**);

void parse_arguments(int, char**);
void printstuff();
void init();

void randomparticles();
void initeventpool();
void loadparticles();
void randommovement();
void initcelllist();
void addtocelllist(particle* p);
void step();
void findcollision(particle*, particle*);
void findallcollisions();
void findcollisions(particle*, particle*);
void findcollisioncell(particle*, int);
void findcrossing(particle* part);
void collision(event*);
void fakecollision(event*);
void addfakecollision(particle*);
void findlistupdate(particle*);
void cellcross(event*);
void addevent (event*);
void createevent (double time, particle* p1, particle* p2, int type);
void addnexteventlist();

void removeevent (event*);
void advance_particles_to_sim_time();
void dump_particles();
double random_gaussian();
void apply_periodic_boundaries(particle* p);
void prepare_dump_file_paths();

const char *event_type_to_str(int et) {
	switch(et) {
		case WRITE: return "WRITE";
		case COLLISION: return "COLLISION";
		case ROOT: return "ROOT";
		case XMINUS: return "X-";
		case XPLUS: return "X+";
		case YMINUS: return "Y-";
		case YPLUS: return "Y+";
		case ZMINUS: return "Z-";
		case ZPLUS: return "Z+";
		default: return "UNKNOWN";
	}
}