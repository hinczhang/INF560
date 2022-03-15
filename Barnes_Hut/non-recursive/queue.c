#include "queue.h"

Queue* CreateQueue() {
    Queue* q = (Queue*)malloc(sizeof(Queue));
    if (!q) {
        printf("SPACE NOT ENOUGH\n");
        return NULL;
    }
    q->front = -1;
    q->rear = -1;
    q->size = 0;
    return q;
}
 
int IsFullQ(Queue* q) {
    return (q->size == MAXSIZE);
}
 
void AddQ(Queue* q, ElementType item) {
    if (IsFullQ(q)) {
        printf("QUEUE FULL\n");
        return;
    }
    q->rear++;
    q->rear %= MAXSIZE;
    q->size++;
    q->data[q->rear] = item;
}
 
int IsEmptyQ(Queue* q) {
    return (q->size == 0);
}
 
ElementType DeleteQ(Queue* q) {
    if (IsEmptyQ(q)) {
        printf("NON_ELEMENT\n");
        node_t *error = NULL;
        return error;
    }
    q->front++;
    q->front %= MAXSIZE; //0 1 2 3 4 5
    q->size--;
    return q->data[q->front];
}
 

